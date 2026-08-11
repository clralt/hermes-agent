from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        yield conn


def _git_repo(path: Path) -> tuple[str, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "done.txt").write_text("phase-1\n", encoding="utf-8")
    subprocess.run(["git", "add", "done.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=path, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=path, text=True).strip()
    return commit, tree


def _checkpoint(repo: Path, *, role: str = "implementer", remaining=None, complete=False):
    from hermes_cli.resumable_execution import capture_repository_state

    state = capture_repository_state(repo)
    return {
        "schema_version": 1,
        "task_id": "task-1",
        "task_version": "v1",
        "task_objective": "complete an oversized governed task",
        "authorized_scope": {"repository": str(repo), "write_paths": ["done.txt", "next.txt"]},
        "execution_role": role,
        "worker": {"identity": f"worker-{role}", "model": "gpt-5.6-sol", "reasoning_effort": "high"},
        "current_phase": "implementation",
        "phase_status": "COMPLETED" if complete else "IN_PROGRESS",
        "work_complete": complete,
        "base_commit": state["head_commit"],
        "base_tree": state["head_tree"],
        "current_candidate_state": state,
        "bindings": {
            "candidate_tree": state["candidate_tree"],
            "dossier_sha256": None,
            "contract_sha256": "a" * 64,
            "evidence_sha256": ["b" * 64],
        },
        "completed_steps": ["write done.txt"],
        "remaining_steps": list(remaining if remaining is not None else ["write next.txt", "validate"]),
        "artifacts_produced": [{"path": "done.txt"}],
        "validation_performed": [{"command": "test -f done.txt", "status": "PASS"}],
        "validation_required": ["pytest"],
        "audit": {
            "status": "NOT_STARTED",
            "reviewer_identity": None,
            "model": None,
            "reasoning_effort": None,
            "independence_provenance": None,
            "completed_criteria": [],
            "remaining_criteria": [],
        },
        "unresolved_findings": [],
        "decisions": ["preserve phase-1 artifact"],
        "external_actions": [],
        "continuation_instruction": "verify repository state, then write next.txt",
        "retry_count": 0,
        "timestamps": {"created_at_utc": "2026-08-09T00:00:00Z", "updated_at_utc": "2026-08-09T00:00:00Z"},
        "provenance": {"source_run_id": 1, "yield_reason": "ITERATION_CAP_REACHED"},
    }


def test_mid_implementation_exhaustion_enqueues_one_durable_continuation(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="oversized", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    assert kb.claim_task(board, tid, claimer="worker-1")
    run_id = kb.get_task(board, tid).current_run_id

    cp = _checkpoint(repo)
    cp["task_id"] = tid
    first = kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    replay = kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)

    assert first.id == replay.id
    task = kb.get_task(board, tid)
    assert task.status == "ready"
    assert task.work_item_kind == "continuation"
    assert task.continuation_count == 1
    assert kb.get_latest_continuation_checkpoint(board, tid).remaining_steps == ["write next.txt", "validate"]
    assert board.execute("SELECT count(*) FROM continuation_checkpoints WHERE task_id=?", (tid,)).fetchone()[0] == 1


def test_mid_validation_checkpoint_preserves_completed_and_remaining_checks(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="validate", assignee="validator", workspace_kind="dir", workspace_path=str(repo))
    kb.claim_task(board, tid, claimer="validator-1")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo, role="validator", remaining=["integration suite", "secret scan"])
    cp["task_id"] = tid
    cp["current_phase"] = "validation"
    cp["validation_performed"] = [{"command": "unit suite", "status": "PASS"}]
    cp["validation_required"] = ["integration suite", "secret scan"]

    kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    restored = kb.get_latest_continuation_checkpoint(board, tid).payload
    assert restored["validation_performed"] == [{"command": "unit suite", "status": "PASS"}]
    assert restored["validation_required"] == ["integration suite", "secret scan"]
    assert restored["work_complete"] is False


def test_repeated_exhaustion_chains_without_manual_restart(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="four runs", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    for continuation in range(1, 4):
        kb.claim_task(board, tid, claimer=f"worker-{continuation}")
        run_id = kb.get_task(board, tid).current_run_id
        cp = _checkpoint(repo, remaining=[f"phase-{continuation + 1}"])
        cp["task_id"] = tid
        cp["retry_count"] = continuation - 1
        cp["provenance"]["source_run_id"] = run_id
        kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    assert kb.get_task(board, tid).continuation_count == 3
    assert [c.continuation_number for c in kb.list_continuation_checkpoints(board, tid)] == [1, 2, 3]


def test_state_divergence_fails_closed_before_restore(board, tmp_path):
    from hermes_cli.resumable_execution import StateDivergenceError, verify_repository_state

    repo = tmp_path / "repo"
    _git_repo(repo)
    cp = _checkpoint(repo)
    (repo / "done.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(StateDivergenceError):
        verify_repository_state(repo, cp["current_candidate_state"])


def test_repository_checkpoint_restores_bound_tracked_untracked_and_explicit_ignored_bytes(
    tmp_path,
):
    from hermes_cli.resumable_execution import (
        capture_repository_state,
        restore_repository_state,
        verify_repository_state,
    )

    repo = tmp_path / "restorable"
    _git_repo(repo)
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked-before\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("ignored-before\n", encoding="utf-8")
    state = capture_repository_state(repo, ignored_paths=["ignored.txt"])

    (repo / "done.txt").write_text("tracked-after\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked-after\n", encoding="utf-8")
    (repo / "ignored.txt").unlink()

    restored = restore_repository_state(repo, state)

    assert (repo / "done.txt").read_text(encoding="utf-8") == "phase-1\n"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "untracked-before\n"
    assert (repo / "ignored.txt").read_text(encoding="utf-8") == "ignored-before\n"
    assert restored == verify_repository_state(repo, state)


def test_production_checkpoint_configuration_includes_explicit_ignored_bytes(
    tmp_path, monkeypatch,
):
    from hermes_cli.resumable_execution import capture_repository_state

    repo = tmp_path / "configured-ignored"
    _git_repo(repo)
    (repo / ".gitignore").write_text("durable-cache/\n", encoding="utf-8")
    (repo / "durable-cache").mkdir()
    (repo / "durable-cache" / "state.bin").write_bytes(b"durable-state")
    monkeypatch.setenv(
        "HERMES_RESUMABLE_IGNORED_PATHS", '["durable-cache"]',
    )

    state = capture_repository_state(repo)

    assert state["repository_inclusion_policy"]["explicit_ignored_paths"] == [
        "durable-cache",
    ]


def test_repository_restore_refuses_directory_collision_with_ignored_data(tmp_path):
    from hermes_cli.resumable_execution import (
        StateDivergenceError,
        capture_repository_state,
        restore_repository_state,
    )

    repo = tmp_path / "directory-collision"
    _git_repo(repo)
    (repo / ".gitignore").write_text("done.txt/cache/\n", encoding="utf-8")
    state = capture_repository_state(repo)
    (repo / "done.txt").unlink()
    (repo / "done.txt" / "cache").mkdir(parents=True)
    (repo / "done.txt" / "cache" / "state.bin").write_bytes(b"do-not-delete")

    with pytest.raises(StateDivergenceError, match="directory collision"):
        restore_repository_state(repo, state)
    assert (repo / "done.txt" / "cache" / "state.bin").read_bytes() == b"do-not-delete"


def test_repository_snapshot_materializes_non_writable_audit_workspace(tmp_path):
    from hermes_cli.resumable_execution import (
        capture_repository_state,
        materialize_repository_snapshot,
    )

    repo = tmp_path / "audit-source"
    _git_repo(repo)
    (repo / "untracked.txt").write_text("candidate\n", encoding="utf-8")
    state = capture_repository_state(repo)
    destination = tmp_path / "audit-materialized"

    materialize_repository_snapshot(state, destination)

    assert (destination / "done.txt").read_text(encoding="utf-8") == "phase-1\n"
    assert (destination / "untracked.txt").read_text(encoding="utf-8") == "candidate\n"
    assert destination.stat().st_mode & 0o222 == 0
    assert (destination / "done.txt").stat().st_mode & 0o222 == 0


def test_repository_checkpoint_refuses_to_delete_unbound_untracked_bytes(tmp_path):
    from hermes_cli.resumable_execution import (
        StateDivergenceError,
        capture_repository_state,
        restore_repository_state,
    )

    repo = tmp_path / "safe-restore"
    _git_repo(repo)
    state = capture_repository_state(repo)
    (repo / "arrived-later.txt").write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(StateDivergenceError, match="unbound untracked"):
        restore_repository_state(repo, state)
    assert (repo / "arrived-later.txt").read_text(encoding="utf-8") == "do not delete\n"


def test_spawn_rejects_changed_worker_identity_at_continuation_boundary(
    board, tmp_path,
):
    repo = tmp_path / "identity-bound"
    _git_repo(repo)
    tid = kb.create_task(board, title="bound", assignee="original-worker",
        workspace_kind="dir", workspace_path=str(repo))
    first = kb.claim_task(board, tid, claimer="first")
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = first.current_run_id
    cp["worker"]["identity"] = "different-worker"
    kb.yield_task_for_continuation(
        board, tid, checkpoint=cp, expected_run_id=first.current_run_id,
    )
    resumed = kb.claim_task(board, tid, claimer="second")
    with pytest.raises(RuntimeError, match="identity"):
        kb._default_spawn(resumed, str(repo))


@pytest.mark.parametrize(
    ("profile_config", "diagnostic"),
    [
        ("model:\n  reasoning_effort: high\n", "effective model is unresolved"),
        ("model:\n  default: gpt-5.6-sol\n", "reasoning effort is unresolved"),
    ],
)
def test_continuation_spawn_rejects_unresolved_profile_provenance(
    board, tmp_path, monkeypatch, profile_config, diagnostic
):
    repo = tmp_path / diagnostic.replace(" ", "-")
    _git_repo(repo)
    profile = Path(os.environ["HERMES_HOME"]) / "profiles" / "worker-implementer"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(profile_config, encoding="utf-8")
    tid = kb.create_task(
        board,
        title="unresolved provenance",
        assignee="worker-implementer",
        workspace_kind="dir",
        workspace_path=str(repo),
    )
    first = kb.claim_task(board, tid, claimer="first")
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = first.current_run_id
    kb.yield_task_for_continuation(
        board, tid, checkpoint=cp, expected_run_id=first.current_run_id,
    )
    resumed = kb.claim_task(board, tid, claimer="second")
    monkeypatch.setattr(
        kb,
        "_resolve_hermes_argv",
        lambda: (_ for _ in ()).throw(AssertionError("spawn path reached")),
    )

    with pytest.raises(RuntimeError, match=diagnostic):
        kb._default_spawn(resumed, str(repo))


def test_continuation_launch_is_gated_until_worker_pid_is_durable(
    board, tmp_path, monkeypatch,
):
    repo = tmp_path / "launch-gated"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="gated", assignee="worker-implementer",
        workspace_kind="dir", workspace_path=str(repo),
        provider_override="openai-codex", model_override="gpt-5.6-sol",
        reasoning_effort="high",
    )
    first = kb.claim_task(board, tid, claimer="first")
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = first.current_run_id
    kb.yield_task_for_continuation(
        board, tid, checkpoint=cp, expected_run_id=first.current_run_id,
    )
    resumed = kb.claim_task(board, tid, claimer="second")
    captured = {}

    class FakeProc:
        pid = 424242

        def terminate(self):
            captured["terminated"] = True

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_popen(cmd, **kwargs)
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert kb._default_spawn(resumed, str(repo)) == 424242
    assert captured["cmd"][0] == sys.executable
    assert "os.read" in captured["cmd"][2]
    assert captured["kwargs"]["pass_fds"]
    assert kb.get_task(board, tid).worker_pid == 424242


def test_worker_pid_registration_is_real_active_run_cas(board, tmp_path):
    repo = tmp_path / "pid-cas"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="pid cas", assignee="coder",
        workspace_kind="dir", workspace_path=str(repo),
    )
    claimed = kb.claim_task(board, tid, claimer="worker-a")
    assert claimed is not None
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL "
            "WHERE id=?",
            (tid,),
        )
    with pytest.raises(RuntimeError, match="lost active-run ownership"):
        kb._set_worker_pid(
            board, tid, 424242,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
        )
    row = board.execute(
        "SELECT status, current_run_id, worker_pid FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert dict(row) == {"status": "ready", "current_run_id": None, "worker_pid": None}


def test_worker_pid_registration_rejects_different_pid_for_same_run(board, tmp_path):
    repo = tmp_path / "pid-cas-rebind"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="pid cas rebind", assignee="coder",
        workspace_kind="dir", workspace_path=str(repo),
    )
    claimed = kb.claim_task(board, tid, claimer="worker-a")
    assert claimed is not None
    kb._set_worker_pid(
        board, tid, 424242,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
    )
    with pytest.raises(RuntimeError, match="different worker PID"):
        kb._set_worker_pid(
            board, tid, 434343,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
        )
    row = board.execute(
        "SELECT worker_pid FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["worker_pid"] == 424242


def test_governed_execution_identity_forces_independent_audit_policy(board):
    assignee = "authorized-exact-tree-closer"
    tid = kb.create_task(
        board, title="governed identity", assignee=assignee,
        requires_independent_audit=False,
    )
    assert kb.get_task(board, tid).requires_independent_audit is True

    ordinary = kb.create_task(board, title="reassigned governed identity")
    assert kb.assign_task(board, ordinary, assignee)
    assert kb.get_task(board, ordinary).requires_independent_audit is True


def test_fresh_launch_is_gated_until_worker_pid_is_durable(board, tmp_path, monkeypatch):
    repo = tmp_path / "fresh-launch-gated"
    _git_repo(repo)
    tid = kb.create_task(
        board,
        title="fresh gated",
        assignee="worker-implementer",
        workspace_kind="dir",
        workspace_path=str(repo),
    )
    claimed = kb.claim_task(board, tid, claimer="fresh")
    captured = {}

    class FakeProc:
        pid = 434343

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            captured["waited"] = timeout

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_popen(cmd, **kwargs)
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert kb._default_spawn(claimed, str(repo)) == 434343
    assert captured["cmd"][0] == sys.executable
    assert "os.read" in captured["cmd"][2]
    assert captured["kwargs"]["pass_fds"]
    task = kb.get_task(board, tid)
    run = kb.latest_run(board, tid)
    assert task.worker_pid == 434343
    assert run.worker_pid == 434343


def test_fresh_launch_pid_cas_failure_never_releases_payload(
    board, tmp_path, monkeypatch,
):
    repo = tmp_path / "fresh-launch-cas"
    _git_repo(repo)
    tid = kb.create_task(
        board,
        title="fresh cas",
        assignee="worker-implementer",
        workspace_kind="dir",
        workspace_path=str(repo),
    )
    claimed = kb.claim_task(board, tid, claimer="fresh")
    captured = {}

    class FakeProc:
        pid = 454545

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            captured["waited"] = timeout

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: FakeProc())
    monkeypatch.setattr(kb, "_set_worker_pid", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("stale run")))

    with pytest.raises(ValueError, match="stale run"):
        kb._default_spawn(claimed, str(repo))
    assert captured["terminated"] is True
    assert "waited" in captured


def test_continuation_spawn_restores_bound_bytes_before_releasing_worker(
    board, tmp_path, monkeypatch,
):
    repo = tmp_path / "spawn-restores"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="restore", assignee="worker-implementer",
        workspace_kind="dir", workspace_path=str(repo),
        provider_override="openai-codex", model_override="gpt-5.6-sol",
        reasoning_effort="high",
    )
    first = kb.claim_task(board, tid, claimer="first")
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = first.current_run_id
    kb.yield_task_for_continuation(
        board, tid, checkpoint=cp, expected_run_id=first.current_run_id,
    )
    (repo / "done.txt").write_text("drifted\n", encoding="utf-8")
    resumed = kb.claim_task(board, tid, claimer="second")

    class FakeProc:
        pid = 464646
        def terminate(self):
            pass
        def wait(self, timeout=None):
            pass

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return real_popen(cmd, **kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert kb._default_spawn(resumed, str(repo)) == 464646
    assert (repo / "done.txt").read_text(encoding="utf-8") == "phase-1\n"


def test_reclaimed_prelaunch_run_rebinds_same_checkpoint_idempotently(board, tmp_path):
    repo = tmp_path / "reclaim"
    _git_repo(repo)
    tid = kb.create_task(board, title="reclaim", assignee="worker-implementer",
        workspace_kind="dir", workspace_path=str(repo))
    first = kb.claim_task(board, tid, claimer="first")
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = first.current_run_id
    saved = kb.yield_task_for_continuation(
        board, tid, checkpoint=cp, expected_run_id=first.current_run_id,
    )
    second = kb.claim_task(board, tid, claimer="second")
    assert kb.get_latest_continuation_checkpoint(board, tid).disposition == "resumed"
    assert kb.reclaim_task(board, tid, reason="dispatcher died before launch")
    pending = kb.get_latest_continuation_checkpoint(board, tid)
    assert pending.id == saved.id
    assert pending.disposition == "pending"
    third = kb.claim_task(board, tid, claimer="third")
    rebound = kb.get_latest_continuation_checkpoint(board, tid)
    assert rebound.id == saved.id
    assert rebound.resumed_run_id == third.current_run_id
    assert third.current_run_id != second.current_run_id


def test_no_work_left_does_not_enqueue_continuation(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="already done", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    kb.claim_task(board, tid, claimer="worker")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo, remaining=[], complete=True)
    cp["task_id"] = tid
    with pytest.raises(ValueError, match="work_complete"):
        kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    assert kb.get_task(board, tid).status == "running"


def test_unsigned_model_audit_cannot_grant_closer_authority():
    checkpoint = {
        "task_id": "close-1",
        "bindings": {
            "candidate_tree": "a" * 40,
            "contract_sha256": "b" * 64,
            "evidence_manifest_sha256": "c" * 64,
            "dossier_sha256": "d" * 64,
        },
        "audit": {
            "status": "CLEAN",
            "reviewer_identity": "codex-high-independent-auditor",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "authority_provenance": {
                "audit_task_id": "audit-1",
                "checkpoint_sha256": "e" * 64,
            },
        },
    }
    with pytest.raises(ValueError, match="signed audit verdict"):
        kb._verify_signed_closer_verdict(checkpoint)


def test_worker_owned_trust_store_cannot_repin_audit_authority(tmp_path, monkeypatch):
    from hermes_cli import config

    trust_path = tmp_path / "attacker-trust.json"
    trust_bytes = b'{"keys": {}}\n'
    trust_path.write_bytes(trust_bytes)
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "kanban": {
                "audit_trust_store_path": str(trust_path),
                "audit_trust_store_sha256": hashlib.sha256(trust_bytes).hexdigest(),
            },
        },
    )
    checkpoint = {
        "audit": {"signed_verdict": {"signature": "attacker-selected"}},
    }

    with pytest.raises(ValueError, match="operator-owned"):
        kb._verify_signed_closer_verdict(checkpoint)


def test_governed_completion_has_no_caller_controlled_bypass(board, tmp_path):
    repo = tmp_path / "no-bypass"
    _git_repo(repo)
    tid = kb.create_task(
        board,
        title="governed",
        assignee="closer",
        workspace_kind="dir",
        workspace_path=str(repo),
        requires_independent_audit=True,
    )
    kb.claim_task(board, tid, claimer="closer")

    with pytest.raises(TypeError, match="_authorized_terminal_close"):
        kb.complete_task(board, tid, _authorized_terminal_close=True)
    # Even a stranded or directly modified terminal marker is not authority.
    # A matching verified signed-verdict receipt must have been accepted first.
    board.execute(
        "UPDATE tasks SET terminal_state='COMPLETED_CLEAN' WHERE id=?", (tid,)
    )
    board.commit()
    with pytest.raises(ValueError, match="trusted CLEAN"):
        kb.complete_task(board, tid)
    assert kb.get_task(board, tid).status == "running"


def test_auditor_and_closer_role_separation_is_fail_closed(tmp_path):
    from hermes_cli.resumable_execution import validate_checkpoint

    repo = tmp_path / "repo"
    _git_repo(repo)
    cp = _checkpoint(repo, role="auditor")
    cp["worker"] = {"identity": "implementer-1", "model": "gpt-5.6-sol", "reasoning_effort": "high"}
    cp["audit"] = {
        "status": "IN_PROGRESS",
        "reviewer_identity": "implementer-1",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "independence_provenance": {"implemented_candidate": True, "read_only": False},
        "completed_criteria": ["scope"],
        "remaining_criteria": ["semantics"],
    }
    with pytest.raises(ValueError, match="independent auditor"):
        validate_checkpoint(cp)

    cp["worker"]["identity"] = "codex-high-independent-auditor"
    cp["audit"]["reviewer_identity"] = "codex-high-independent-auditor"
    cp["audit"]["independence_provenance"] = {
        "implemented_candidate": False, "read_only": True,
        "fresh_invocation": True, "producer_identity": "implementer-1",
    }
    validate_checkpoint(cp)

    closer = json.loads(json.dumps(cp))
    closer["execution_role"] = "closer"
    closer["worker"]["identity"] = "authorized-exact-tree-closer"
    closer["audit"]["status"] = "CLEAN"
    closer["bindings"]["audited_candidate_tree"] = closer["bindings"]["candidate_tree"]
    with pytest.raises(ValueError, match="trusted approved-reviewer authority"):
        validate_checkpoint(closer)
    closer["audit"]["authority_provenance"] = {
        "audit_task_id": "t_independent_audit",
        "audit_run_id": 41,
        "checkpoint_sha256": "c" * 64,
    }
    validate_checkpoint(closer)
    closer["bindings"]["candidate_tree"] = "f" * 40
    with pytest.raises(ValueError, match="exact audited candidate"):
        validate_checkpoint(closer)


def test_mid_remediation_preserves_unresolved_audit_findings(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="remediate", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    kb.claim_task(board, tid, claimer="producer")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo, remaining=["fix finding F-2", "rerun validation"])
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    cp["current_phase"] = "remediation"
    cp["unresolved_findings"] = [
        {"id": "F-2", "status": "PARTIALLY_REMEDIATED", "remaining": "negative test"}
    ]
    cp["audit"]["status"] = "REJECTED"
    kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    restored = kb.get_latest_continuation_checkpoint(board, tid).payload
    assert restored["current_phase"] == "remediation"
    assert restored["unresolved_findings"][0]["id"] == "F-2"


def test_mid_audit_restores_same_exact_read_only_codex_high_role(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="audit", assignee="codex-high-independent-auditor",
        workspace_kind="dir", workspace_path=str(repo),
        provider_override="codex", model_override="gpt-5.6-sol",
        reasoning_effort="high",
    )
    kb.claim_task(board, tid, claimer="audit-run-1")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo, role="auditor", remaining=["audit semantics", "emit verdict"])
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    cp["current_phase"] = "independent-audit"
    cp["worker"] = {
        "identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }
    cp["audit"] = {
        "status": "IN_PROGRESS",
        "reviewer_identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "independence_provenance": {
            "implemented_candidate": False, "read_only": True,
            "fresh_invocation": True, "producer_identity": "producer-worker",
        },
        "completed_criteria": ["scope", "provenance"],
        "remaining_criteria": ["semantics", "negative tests"],
    }
    tree = cp["bindings"]["candidate_tree"]
    kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)
    restored = kb.get_latest_continuation_checkpoint(board, tid).payload
    assert restored["worker"] == cp["worker"]
    assert restored["bindings"]["candidate_tree"] == tree
    assert restored["audit"]["completed_criteria"] == ["scope", "provenance"]
    assert kb.get_task(board, tid).status == "ready"  # no premature commit/close


def test_auditor_cap_without_rich_heartbeat_still_gets_safe_emergency_checkpoint(
    tmp_path, monkeypatch,
):
    from hermes_cli.resumable_execution import (
        build_emergency_checkpoint, validate_checkpoint,
    )

    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.setenv("HERMES_EXECUTION_ROLE", "auditor")
    monkeypatch.setenv("HERMES_WORKER_IDENTITY", "codex-high-independent-auditor")
    task = SimpleNamespace(
        id="t_audit", created_at=1, title="audit", body=None,
        assignee="codex-high-independent-auditor", workspace_kind="dir",
        workspace_path=str(repo), current_step_key="independent-audit",
        provider_override="codex", model_override="gpt-5.6-sol",
        reasoning_effort="high",
    )
    cp = build_emergency_checkpoint(
        task=task, run_id=9, agent=SimpleNamespace(model="wrong-default"),
        summary="iteration cap", previous=None,
    )
    validate_checkpoint(cp)
    assert cp["worker"] == {
        "identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol", "reasoning_effort": "high",
    }
    assert cp["audit"]["completed_criteria"] == []
    assert cp["audit"]["remaining_criteria"]
    # An emergency controller checkpoint preserves resumability but must not
    # fabricate independent/read-only/fresh provenance on the auditor's behalf.
    assert cp["audit"]["independence_provenance"] is None


def test_closer_requires_prior_completed_trusted_audit_checkpoint(
    board, tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.setattr(
        kb, "_verify_signed_closer_verdict",
        lambda *_args, **_kwargs: {
            "verdict_id": "trusted-test-verdict",
            "verdict_sha256": "9" * 64,
            "status": "CLEAN",
        },
    )
    audit_tid = kb.create_task(
        board, title="independent audit", assignee="desktop-fable-5-read-only",
        workspace_kind="dir", workspace_path=str(repo),
        provider_override="fable", model_override="claude-fable-5",
        reasoning_effort="none",
    )
    audit_task = kb.claim_task(board, audit_tid, claimer="fresh-audit-invocation")
    audit_run = audit_task.current_run_id
    assert audit_run is not None
    audit_cp = _checkpoint(repo, role="auditor", remaining=[])
    audit_cp["task_id"] = audit_tid
    audit_cp["provenance"]["source_run_id"] = audit_run
    audit_cp["worker"] = {
        "identity": "desktop-fable-5-read-only", "model": "claude-fable-5",
        "reasoning_effort": "none",
    }
    audit_cp["audit"] = {
        "status": "CLEAN", "reviewer_identity": "desktop-fable-5-read-only",
        "model": "claude-fable-5", "reasoning_effort": "none",
        "independence_provenance": {
            "implemented_candidate": False, "read_only": True,
            "fresh_invocation": True, "producer_identity": "producer-worker",
        },
        "completed_criteria": ["all"], "remaining_criteria": [],
    }
    audit_sha = kb.save_phase_checkpoint(
        board, audit_tid, checkpoint=audit_cp, expected_run_id=audit_run,
    )
    assert kb.complete_task(board, audit_tid, result="CLEAN", expected_run_id=audit_run)

    closer_tid = kb.create_task(
        board, title="exact-tree close", assignee="authorized-exact-tree-closer",
        workspace_kind="dir", workspace_path=str(repo),
        requires_independent_audit=True,
    )
    closer_task = kb.claim_task(board, closer_tid, claimer="closer-run")
    closer_run = closer_task.current_run_id
    closer_cp = json.loads(json.dumps(audit_cp))
    closer_cp["task_id"] = closer_tid
    closer_cp["provenance"]["source_run_id"] = closer_run
    closer_cp["execution_role"] = "closer"
    closer_cp["worker"]["identity"] = "authorized-exact-tree-closer"
    closer_cp["bindings"]["audited_candidate_tree"] = closer_cp["bindings"]["candidate_tree"]
    closer_cp["audit"]["authority_provenance"] = {
        "audit_task_id": audit_tid, "audit_run_id": audit_run,
        "checkpoint_sha256": audit_sha,
    }
    with kb.write_txn(board):
        board.execute(
            "UPDATE task_runs SET status='yielded', outcome=NULL WHERE id=?",
            (audit_run,),
        )
    with pytest.raises(ValueError, match="audit run.*completed"):
        kb.save_phase_checkpoint(
            board, closer_tid, checkpoint=closer_cp, expected_run_id=closer_run,
        )
    with kb.write_txn(board):
        board.execute(
            "UPDATE task_runs SET status='done', outcome='completed' WHERE id=?",
            (audit_run,),
        )
    assert kb.save_phase_checkpoint(
        board, closer_tid, checkpoint=closer_cp, expected_run_id=closer_run,
    )
    forged = json.loads(json.dumps(closer_cp))
    forged["audit"]["authority_provenance"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="trusted audited bytes"):
        kb.save_phase_checkpoint(
            board, closer_tid, checkpoint=forged, expected_run_id=closer_run,
        )
    with pytest.raises(ValueError, match="trusted CLEAN"):
        kb.complete_task(
            board, closer_tid, result="implementer bypass", expected_run_id=closer_run,
        )
    closer_cp["work_complete"] = True
    closer_cp["terminal_state"] = "COMPLETED_CLEAN"
    assert kb.apply_terminal_checkpoint(
        board, closer_tid, checkpoint=closer_cp, expected_run_id=closer_run,
        result="exact audited bytes closed",
    ) == "COMPLETED_CLEAN"
    closed = kb.get_task(board, closer_tid)
    assert closed.status == "done"
    assert closed.terminal_state == "COMPLETED_CLEAN"

    # Reopening creates a new run.  The consumed receipt from the prior exact
    # terminal transition must not authorize this new incarnation.
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?",
            (closer_tid,),
        )
    reopened = kb.claim_task(board, closer_tid, claimer="second-run")
    assert reopened is not None and reopened.current_run_id != closer_run
    with pytest.raises(ValueError, match="trusted CLEAN"):
        kb.complete_task(
            board, closer_tid, result="replayed authority",
            expected_run_id=reopened.current_run_id,
        )


def test_operator_exception_is_terminal_only_at_reserved_boundary(board, tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="purchase paid API", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    kb.claim_task(board, tid, claimer="worker")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo, remaining=["purchase paid API plan"])
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    cp["terminal_state"] = "BLOCKED_OPERATOR_EXCEPTION"
    assert kb.apply_terminal_checkpoint(
        board, tid, checkpoint=cp, expected_run_id=run_id,
        result="Current-session spending authority required",
    ) == "BLOCKED_OPERATOR_EXCEPTION"
    task = kb.get_task(board, tid)
    assert task.status == "blocked"
    assert task.terminal_state == "BLOCKED_OPERATOR_EXCEPTION"
    assert kb.list_continuation_checkpoints(board, tid) == []


def test_restart_recovers_saved_phase_checkpoint_without_failure_count(board, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(board, title="restart recovery", assignee="coder", workspace_kind="dir", workspace_path=str(repo))
    claimed = kb.claim_task(board, tid)
    run_id = claimed.current_run_id
    cp = _checkpoint(repo, remaining=["resume after restart"])
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    kb.save_phase_checkpoint(board, tid, checkpoint=cp, expected_run_id=run_id)
    kb._set_worker_pid(board, tid, 999999)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    assert tid not in kb.detect_crashed_workers(board)
    task = kb.get_task(board, tid)
    assert task.status == "ready"
    assert task.work_item_kind == "continuation"
    assert task.consecutive_failures == 0
    assert kb.get_latest_continuation_checkpoint(board, tid).reason == "WORKER_INTERRUPTED"


def test_oversized_fiis_task_completes_across_four_clean_low_cap_invocations(
    board, tmp_path, monkeypatch,
):
    """Dispatcher proof: every automatically spawned subprocess gets one phase."""
    repo = tmp_path / "oversized-fiis"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="Intentionally oversized FIIS continuity proof",
        assignee="fiis-worker", workspace_kind="dir", workspace_path=str(repo),
    )
    db_path = board.execute("PRAGMA database_list").fetchone()[2]
    worker = tmp_path / "bounded_worker.py"
    worker.write_text(
        """
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from hermes_cli import kanban_db as kb
from hermes_cli.resumable_execution import capture_repository_state, verify_repository_state

db, tid, repo, expected_run = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
with kb.connect(db) as conn:
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == 'running'
    run_id = task.current_run_id
    assert run_id == expected_run
    prior = kb.get_latest_continuation_checkpoint(conn, tid)
    completed = list(prior.payload['completed_steps']) if prior else []
    if prior:
        verify_repository_state(repo, prior.payload['current_candidate_state'])
    phase = len(completed) + 1
    target = repo / f'phase-{phase}.txt'
    assert not target.exists(), 'completed phase was repeated'
    target.write_text(f'phase {phase} completed by clean invocation\\n')
    completed.append(f'phase-{phase}')
    if phase == 4:
        assert kb.complete_task(conn, tid, result='four bounded phases complete', expected_run_id=run_id)
        print(json.dumps({'phase': phase, 'run_id': run_id, 'disposition': 'COMPLETED'}))
        raise SystemExit(0)
    state = capture_repository_state(repo)
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    cp = {
      'schema_version': 1, 'task_id': tid, 'task_version': 'v1',
      'task_objective': 'oversized FIIS continuity proof',
      'authorized_scope': {'repository': str(repo), 'write_paths': ['phase-*.txt']},
      'execution_role': 'implementer',
      'worker': {'identity': 'fiis-worker', 'model': 'test-bounded-worker', 'reasoning_effort': 'high'},
      'current_phase': f'implementation-{phase}', 'phase_status': 'IN_PROGRESS',
      'work_complete': False, 'base_commit': state['head_commit'], 'base_tree': state['head_tree'],
      'current_candidate_state': state,
      'bindings': {'candidate_tree': state['candidate_tree'], 'dossier_sha256': None,
                   'contract_sha256': 'a'*64, 'evidence_sha256': []},
      'completed_steps': completed,
      'remaining_steps': [f'phase-{n}' for n in range(phase + 1, 5)],
      'artifacts_produced': [{'path': f'phase-{n}.txt'} for n in range(1, phase + 1)],
      'validation_performed': [], 'validation_required': ['final phase validation'],
      'audit': {'status': 'NOT_STARTED', 'reviewer_identity': None, 'model': None,
                'reasoning_effort': None, 'independence_provenance': None,
                'completed_criteria': [], 'remaining_criteria': []},
      'unresolved_findings': [], 'decisions': [], 'external_actions': [],
      'continuation_instruction': f'verify state and execute phase-{phase + 1}',
      'retry_count': 0, 'timestamps': {'created_at_utc': now, 'updated_at_utc': now},
      'provenance': {'source_run_id': run_id, 'yield_reason': 'ITERATION_CAP_REACHED'},
    }
    saved = kb.yield_task_for_continuation(conn, tid, checkpoint=cp, expected_run_id=run_id)
    print(json.dumps({'phase': phase, 'disposition': 'YIELDED',
                      'run_id': run_id,
                      'checkpoint_id': saved.id,
                      'checkpoint_sha256': saved.checkpoint_sha256,
                      'continuation_number': saved.continuation_number}))
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2])
    outputs = []
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)

    def spawn_bounded_phase(task, workspace):
        proc = subprocess.run(
            [
                sys.executable, str(worker), str(db_path), tid, str(repo),
                str(task.current_run_id),
            ],
            text=True, capture_output=True, env=env, check=True,
        )
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        return None

    for _ in range(4):
        tick = kb.dispatch_once(
            board, spawn_fn=spawn_bounded_phase, max_spawn=1,
            reconcile_orphans=False,
        )
        assert [item[0] for item in tick.spawned] == [tid]
    assert [item["disposition"] for item in outputs] == [
        "YIELDED", "YIELDED", "YIELDED", "COMPLETED",
    ]
    assert [item.get("continuation_number") for item in outputs[:3]] == [1, 2, 3]
    assert len({item["run_id"] for item in outputs}) == 4
    assert all(len(item["checkpoint_sha256"]) == 64 for item in outputs[:3])
    task = kb.get_task(board, tid)
    assert task.status == "done"
    assert task.continuation_count == 3
    assert [p.name for p in sorted(repo.glob("phase-*.txt"))] == [
        "phase-1.txt", "phase-2.txt", "phase-3.txt", "phase-4.txt",
    ]


def test_role_separated_pipeline_advances_after_exact_tree_close(
    board, tmp_path, monkeypatch,
):
    """Implementation -> validation -> resumable audit -> closer -> next task."""
    from hermes_cli.resumable_execution import checkpoint_sha256

    repo = tmp_path / "pipeline"
    _git_repo(repo)
    monkeypatch.setattr(
        kb, "_verify_signed_closer_verdict",
        lambda *_args, **_kwargs: {
            "verdict_id": "trusted-pipeline-verdict",
            "verdict_sha256": "8" * 64,
            "status": "CLEAN",
        },
    )

    def cp_for(task_id, run_id, role="implementer", remaining=None):
        cp = _checkpoint(repo, role=role, remaining=remaining)
        cp["task_id"] = task_id
        cp["provenance"]["source_run_id"] = run_id
        return cp

    impl = kb.create_task(board, title="implement", assignee="implementer",
        workspace_kind="dir", workspace_path=str(repo))
    validate = kb.create_task(board, title="validate", assignee="validator", parents=(impl,),
        workspace_kind="dir", workspace_path=str(repo))
    audit = kb.create_task(board, title="audit", assignee="codex-high-independent-auditor",
        parents=(validate,), workspace_kind="dir", workspace_path=str(repo),
        work_item_kind="audit", provider_override="codex",
        model_override="gpt-5.6-sol", reasoning_effort="high")
    close = kb.create_task(board, title="close", assignee="authorized-exact-tree-closer",
        parents=(audit,), workspace_kind="dir", workspace_path=str(repo),
        requires_independent_audit=True)
    next_task = kb.create_task(board, title="next", assignee="next", parents=(close,),
        workspace_kind="dir", workspace_path=str(repo))

    run1 = kb.claim_task(board, impl, claimer="impl-1").current_run_id
    kb.yield_task_for_continuation(
        board, impl, checkpoint=cp_for(impl, run1), expected_run_id=run1,
    )
    run2 = kb.claim_task(board, impl, claimer="impl-2").current_run_id
    assert kb.complete_task(board, impl, result="implemented", expected_run_id=run2)
    assert kb.get_task(board, validate).status == "ready"

    val_run = kb.claim_task(board, validate, claimer="validator").current_run_id
    assert kb.complete_task(board, validate, result="validated", expected_run_id=val_run)
    assert kb.get_task(board, audit).status == "ready"

    audit_run1 = kb.claim_task(board, audit, claimer="audit-1").current_run_id
    partial = cp_for(audit, audit_run1, "auditor", ["semantic audit"])
    partial["worker"] = {"identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol", "reasoning_effort": "high"}
    partial["audit"] = {"status": "IN_PROGRESS",
        "reviewer_identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol", "reasoning_effort": "high",
        "independence_provenance": {"fresh_invocation": True, "read_only": True,
            "implemented_candidate": False, "producer_identity": "implementer"},
        "completed_criteria": ["scope"], "remaining_criteria": ["semantics"]}
    kb.yield_task_for_continuation(
        board, audit, checkpoint=partial, expected_run_id=audit_run1,
    )
    audit_run2 = kb.claim_task(board, audit, claimer="audit-2").current_run_id
    clean = cp_for(audit, audit_run2, "auditor", [])
    clean["worker"] = dict(partial["worker"])
    clean["audit"] = dict(partial["audit"])
    clean["audit"].update({"status": "CLEAN",
        "completed_criteria": ["scope", "semantics"], "remaining_criteria": []})
    kb.save_phase_checkpoint(board, audit, checkpoint=clean, expected_run_id=audit_run2)
    audit_sha = checkpoint_sha256(clean)
    assert kb.complete_task(board, audit, result="CLEAN", expected_run_id=audit_run2)
    assert kb.get_task(board, close).status == "ready"

    close_run = kb.claim_task(board, close, claimer="closer").current_run_id
    closer = cp_for(close, close_run, "closer", [])
    closer["worker"]["identity"] = "authorized-exact-tree-closer"
    closer["work_complete"] = True
    closer["terminal_state"] = "COMPLETED_CLEAN"
    closer["audit"] = dict(clean["audit"])
    closer["audit"].update({"source_task_id": audit, "source_run_id": audit_run2,
        "source_checkpoint_sha256": audit_sha,
        "authority_provenance": {"audit_task_id": audit, "audit_run_id": audit_run2,
            "checkpoint_sha256": audit_sha}})
    closer["bindings"]["audited_candidate_tree"] = closer["bindings"]["candidate_tree"]
    closer["authority_provenance"] = {"closer_identity": "authorized-exact-tree-closer",
        "authority_source": "frozen task contract"}
    assert kb.apply_terminal_checkpoint(
        board, close, checkpoint=closer, expected_run_id=close_run,
    ) == "COMPLETED_CLEAN"
    assert kb.get_task(board, next_task).status == "ready"


# ---------------------------------------------------------------------------
# Revision-5 adversarial regressions for signed FAIL findings B-003..B-007.
# Each fails on frozen candidate d5ea5d79 and passes on revision 5.
# ---------------------------------------------------------------------------


def test_restore_write_path_refuses_symlinked_ancestor(tmp_path, monkeypatch):
    """B-003: every restoration write must be symlink-ancestor safe on its own.

    Failure mechanism on candidate d5ea5d79: after lexical ``_safe_relpath``
    validation, restoration performed ``mkdir(exist_ok)`` / ``unlink`` /
    ``mkstemp`` / ``os.replace`` on ``repo / rel`` directly. If a bound path's
    parent directory ``sub`` was a symlink to an external directory at write
    time, all of those operated in the symlink destination OUTSIDE the
    repository — deleting and overwriting the external target.

    The coarse unbound-untracked-paths pre-scan is a set-membership check, not a
    symlink control, and is subject to TOCTOU: it snapshots the path set, then
    the writes happen later. This test isolates the write path from that pre-scan
    (holding the pre-scan's view constant) so the write primitives themselves are
    exercised. The fix walks every ancestor descriptor-relative with O_NOFOLLOW
    (or rejects symlink ancestors where dir_fd is unavailable) and fails closed
    before any read, unlink, mkdir, tempfile creation, or replace.
    """
    from hermes_cli import resumable_execution as rex
    from hermes_cli.resumable_execution import (
        StateDivergenceError, capture_repository_state, restore_repository_state,
    )

    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "sub").mkdir()
    (repo / "sub" / "file.txt").write_text("bound-content\n", encoding="utf-8")
    state = capture_repository_state(repo)

    external = tmp_path / "external"
    external.mkdir()
    victim = external / "file.txt"
    victim.write_text("EXTERNAL-ORIGINAL\n", encoding="utf-8")

    # Freeze the pre-scan's view to the pre-swap path set so the unbound-paths
    # guard cannot mask the write-path defect (models the real TOCTOU window
    # between the scan and the writes). The write primitives must stand on their
    # own.
    clean_paths, clean_explicit = rex._included_paths(repo, ())
    monkeypatch.setattr(
        rex, "_included_paths", lambda *_a, **_k: (clean_paths, clean_explicit),
    )

    # Attacker swaps the bound path's parent directory for a symlink to an
    # external directory before the writes run.
    import shutil
    shutil.rmtree(repo / "sub")
    os.symlink(str(external), str(repo / "sub"))

    with pytest.raises(StateDivergenceError):
        restore_repository_state(repo, state)
    # The external target must be untouched — no unlink, no overwrite through
    # the symlinked ancestor.
    assert victim.read_text(encoding="utf-8") == "EXTERNAL-ORIGINAL\n"


def test_clean_exit_without_terminal_call_enqueues_production_continuation(
    board, tmp_path, monkeypatch,
):
    """B-004: an ordinary clean worker exit without kanban_complete/kanban_block
    must persist a durable checkpoint and atomically requeue the task.

    Failure mechanism on candidate d5ea5d79: a checkpoint was only persisted on
    the iteration-cap fallback; a normal clean exit left the task ``running``
    with no machine-readable state and was retried as a protocol violation. The
    forced multi-invocation evidence used a synthetic worker calling
    ``yield_task_for_continuation`` directly and never exercised this path. This
    test drives the real production helper (``build_emergency_checkpoint`` +
    ``yield_task_for_continuation``) that ``finalize_turn`` arms at process exit.
    """
    from types import SimpleNamespace

    from agent import turn_finalizer as tf

    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="clean exit", assignee="coder",
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(board, tid, claimer="worker-1")
    run_id = kb.get_task(board, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.delenv("HERMES_EXECUTION_ROLE", raising=False)

    import logging
    agent = SimpleNamespace(model="coder-model", reasoning_effort="medium")
    applied = tf._yield_kanban_task_checkpoint(
        agent, "All done — here is the report.",
        reason="CLEAN_EXIT_WITHOUT_TERMINAL_CALL",
        logger=logging.getLogger("test"),
    )

    assert applied is True
    task = kb.get_task(board, tid)
    assert task.status == "ready"
    assert task.work_item_kind == "continuation"
    cp = kb.get_latest_continuation_checkpoint(board, tid)
    assert cp is not None
    assert cp.reason == "CLEAN_EXIT_WITHOUT_TERMINAL_CALL"


def test_continuation_launch_rejects_tampered_checkpoint_payload(board, tmp_path):
    """B-005: continuation dispatch must recompute the checkpoint hash and fail
    closed on a payload that no longer matches its immutable digest column.

    Failure mechanism on candidate d5ea5d79: dispatch trusted the DB payload
    without recomputing ``checkpoint_sha256(payload)`` against the immutable
    column, so a tampered payload drove restoration/role selection while the
    advertised hash stayed unchanged.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="tamper", assignee="coder",
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(board, tid, claimer="worker-1")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo)
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)

    # Tamper the stored payload while leaving the immutable checkpoint_sha256
    # column intact (bypassing yield_task_for_continuation's atomic hashing).
    tampered = json.loads(json.dumps(cp))
    tampered["remaining_steps"] = ["attacker-injected step"]
    with kb.write_txn(board):
        board.execute(
            "UPDATE continuation_checkpoints SET payload=? WHERE task_id=?",
            (json.dumps(tampered), tid),
        )

    task = kb.get_task(board, tid)
    with pytest.raises(RuntimeError, match="payload hash"):
        kb._default_spawn(task, str(repo))


def test_continuation_fails_closed_without_resolvable_model_provenance(
    board, tmp_path, monkeypatch,
):
    """B-006: model/effort provenance is enforced even when the task has no
    explicit override but the assignee profile resolves a model — profile-default
    drift at an authority transition must be caught before the subprocess launches.

    Failure mechanism on candidate d5ea5d79: model/effort was checked only when
    task.model_override was non-null; a checkpoint where the profile-default model
    had drifted away from the checkpoint binding was invisible to the guard and
    the continuation launched on the self-declared values.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    tid = kb.create_task(
        board, title="drift", assignee="coder",
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(board, tid, claimer="worker-1")
    run_id = kb.get_task(board, tid).current_run_id
    cp = _checkpoint(repo)  # worker binds gpt-5.6-sol / high
    cp["task_id"] = tid
    cp["provenance"]["source_run_id"] = run_id
    # Identity must match so the check under test is the model/effort provenance,
    # not the identity gate.
    cp["worker"]["identity"] = "coder"
    kb.yield_task_for_continuation(board, tid, checkpoint=cp, expected_run_id=run_id)

    # Simulate a resolvable profile that now resolves to a different model (drift)
    # without the task having an explicit override. On d5ea5d79, no override means
    # no check; on the fixed code the resolved value is compared and the mismatch
    # is caught before any subprocess is launched.
    monkeypatch.setattr(
        kb, "_resolve_effective_model_effort",
        lambda _task: ("profile-drift-model", "high"),
    )

    task = kb.get_task(board, tid)
    with pytest.raises(RuntimeError, match="effective model"):
        kb._default_spawn(task, str(repo))


def test_governed_evidence_kill_is_reachable_through_signed_authority(
    board, tmp_path, monkeypatch,
):
    """B-007: KILLED_BY_EVIDENCE must be a reachable governed terminal state
    through an independently signed authority path.

    Failure mechanism on candidate d5ea5d79: ``apply_terminal_checkpoint``
    inserted a signed authority receipt only on the COMPLETED_CLEAN branch, yet
    ``complete_task`` required an unconsumed receipt for a governed
    KILLED_BY_EVIDENCE completion — so a governed evidence kill was impossible.
    Closer validation also demanded a CLEAN-only audit. The fix gives the kill
    its own signed FAIL authority path.
    """
    from hermes_cli.resumable_execution import checkpoint_sha256

    repo = tmp_path / "kill"
    _git_repo(repo)
    monkeypatch.setattr(
        kb, "_verify_signed_closer_verdict",
        lambda *_a, **_k: {
            "verdict_id": "trusted-kill-verdict",
            "verdict_sha256": "7" * 64,
            "status": "FAIL",
        },
    )

    audit_tid = kb.create_task(
        board, title="independent audit", assignee="codex-high-independent-auditor",
        workspace_kind="dir", workspace_path=str(repo),
        provider_override="codex", model_override="gpt-5.6-sol",
        reasoning_effort="high",
    )
    audit_run = kb.claim_task(board, audit_tid, claimer="fresh-audit").current_run_id
    audit_cp = _checkpoint(repo, role="auditor", remaining=[])
    audit_cp["task_id"] = audit_tid
    audit_cp["provenance"]["source_run_id"] = audit_run
    audit_cp["worker"] = {
        "identity": "codex-high-independent-auditor", "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }
    audit_cp["audit"] = {
        "status": "KILLED_BY_EVIDENCE",
        "reviewer_identity": "codex-high-independent-auditor",
        "model": "gpt-5.6-sol", "reasoning_effort": "high",
        "independence_provenance": {
            "implemented_candidate": False, "read_only": True,
            "fresh_invocation": True, "producer_identity": "producer-worker",
        },
        "completed_criteria": ["all"], "remaining_criteria": [],
    }
    audit_sha = kb.save_phase_checkpoint(
        board, audit_tid, checkpoint=audit_cp, expected_run_id=audit_run,
    )
    assert kb.complete_task(
        board, audit_tid, result="KILLED_BY_EVIDENCE", expected_run_id=audit_run,
    )

    closer_tid = kb.create_task(
        board, title="evidence kill close", assignee="authorized-exact-tree-closer",
        workspace_kind="dir", workspace_path=str(repo),
        requires_independent_audit=True,
    )
    closer_run = kb.claim_task(board, closer_tid, claimer="closer-run").current_run_id
    closer_cp = json.loads(json.dumps(audit_cp))
    closer_cp["task_id"] = closer_tid
    closer_cp["provenance"]["source_run_id"] = closer_run
    closer_cp["execution_role"] = "closer"
    closer_cp["work_complete"] = False
    closer_cp["terminal_state"] = "KILLED_BY_EVIDENCE"
    closer_cp["worker"]["identity"] = "authorized-exact-tree-closer"
    closer_cp["bindings"]["audited_candidate_tree"] = closer_cp["bindings"]["candidate_tree"]
    closer_cp["audit"]["authority_provenance"] = {
        "audit_task_id": audit_tid, "audit_run_id": audit_run,
        "checkpoint_sha256": audit_sha,
    }

    assert kb.apply_terminal_checkpoint(
        board, closer_tid, checkpoint=closer_cp, expected_run_id=closer_run,
        result="rejected by evidence",
    ) == "KILLED_BY_EVIDENCE"
    killed = kb.get_task(board, closer_tid)
    assert killed.status == "done"
    assert killed.terminal_state == "KILLED_BY_EVIDENCE"
