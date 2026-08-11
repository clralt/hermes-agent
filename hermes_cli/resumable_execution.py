"""Durable, role-bound state for resumable Kanban execution.

This module is deliberately independent of conversational history.  It validates
machine-readable phase checkpoints and captures enough Git state for a clean
worker process to fail closed when the workspace diverges before restoration.
"""
from __future__ import annotations

import hashlib
import json
import stat as stat_module
import subprocess
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


REQUIRED_CHECKPOINT_FIELDS = frozenset({
    "schema_version", "task_id", "task_version", "task_objective",
    "authorized_scope", "execution_role", "worker", "current_phase",
    "phase_status", "work_complete", "base_commit", "base_tree",
    "current_candidate_state", "bindings", "completed_steps",
    "remaining_steps", "artifacts_produced", "validation_performed",
    "validation_required", "audit", "unresolved_findings", "decisions",
    "external_actions", "continuation_instruction", "retry_count",
    "timestamps", "provenance",
})
VALID_ROLES = frozenset({"controller", "implementer", "validator", "auditor", "closer"})
VALID_TERMINAL_STATES = frozenset({
    "COMPLETED_CLEAN", "KILLED_BY_EVIDENCE", "BLOCKED_EXTERNAL_DEPENDENCY",
    "BLOCKED_OPERATOR_EXCEPTION", "FAILED_RECOVERABLE",
})
# Ordered policy: Fable 5 is primary; Codex High is the capacity fallback.
# Keep the legacy AUDITOR_* aliases for callers that construct the explicit
# fallback checkpoint, but never use them as the complete accepted-reviewer set.
INDEPENDENT_AUDIT_REVIEWERS = (
    ("desktop-fable-5-read-only", "fable", "claude-fable-5", None),
    ("codex-high-independent-auditor", "codex", "gpt-5.6-sol", "high"),
)
INDEPENDENT_AUDIT_REVIEWER_POLICIES = {
    identity: (provider, model, effort)
    for identity, provider, model, effort in INDEPENDENT_AUDIT_REVIEWERS
}
AUDITOR_IDENTITY = "codex-high-independent-auditor"
AUDITOR_MODEL = "gpt-5.6-sol"
AUDITOR_EFFORT = "high"


def _audit_record_matches_policy(audit: Mapping[str, Any]) -> bool:
    identity = str(audit.get("reviewer_identity") or "")
    policy = INDEPENDENT_AUDIT_REVIEWER_POLICIES.get(identity)
    if policy is None:
        return False
    _provider, model, effort = policy
    actual_effort = audit.get("reasoning_effort")
    return (
        audit.get("model") == model
        and bool(str(actual_effort or "").strip())
        and (effort is None or actual_effort == effort)
    )


class StateDivergenceError(RuntimeError):
    """The live repository no longer equals the checkpoint binding."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def checkpoint_sha256(checkpoint: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest()


def _git(repo: Path, *args: str, allow_failure: bool = False) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if result.returncode and not allow_failure:
        raise ValueError(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _safe_relpath(value: str) -> str:
    rel = str(value).replace("\\", "/").strip("/")
    if not rel or rel == "." or rel.startswith("../") or "/../" in f"/{rel}/":
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    return rel


def execution_role_is_locked() -> bool:
    """True in confined observer processes (auditor / closer).

    Shared predicate for every surface that must not execute extension Python
    (plugins, hooks, middleware) under a confined role. Lives here because this
    module is import-light and already role-aware.
    """
    return os.environ.get("HERMES_EXECUTION_ROLE", "").strip().lower() in {
        "auditor", "closer",
    }


# ---------------------------------------------------------------------------
# Symlink-ancestor-safe filesystem primitives (revision-4 finding B-003).
#
# Lexical validation (``_safe_relpath``) cannot stop a live filesystem attack:
# after validation, replacing a bound path's PARENT directory with a symlink
# makes every subsequent ``repo / rel`` operation (unlink, mkdir, mkstemp,
# os.replace) act inside the symlink's destination, outside the repository.
# Restoration therefore walks each ancestor descriptor-relative with
# O_NOFOLLOW|O_DIRECTORY and performs every mutation via ``dir_fd`` so no
# ancestor component is ever re-resolved through a symlink. On platforms
# without dir_fd support the walk falls back to an lstat rejection of every
# ancestor immediately before each operation (fail closed, best available).
# ---------------------------------------------------------------------------

_O_DIR_NOFOLLOW = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)

_RESTORE_DIRFD_OK = (
    {os.open, os.mkdir, os.rename, os.unlink, os.rmdir, os.stat, os.symlink}
    <= os.supports_dir_fd
    and getattr(os, "O_NOFOLLOW", 0) != 0
    and getattr(os, "O_DIRECTORY", 0) != 0
)


def _reject_symlink_ancestors(repo: Path, rel: str) -> None:
    """Fallback rejection: refuse when any ancestor component is a symlink."""
    current = repo
    for name in _safe_relpath(rel).split("/")[:-1]:
        current = current / name
        if current.is_symlink():
            raise StateDivergenceError(
                f"symlink ancestor in bound path; refusing restore: {rel}"
            )


@contextmanager
def _parent_dirfd(repo: Path, rel: str, *, create_missing: bool) -> Iterator[tuple[int, str]]:
    """Yield ``(dir_fd, leaf_name)`` for the parent of ``repo/rel``.

    Every ancestor component is opened O_NOFOLLOW|O_DIRECTORY relative to the
    previous descriptor, so a symlink introduced anywhere in the chain fails
    with ELOOP/ENOTDIR instead of being followed. Missing intermediate
    directories are created via ``mkdir(dir_fd=...)`` when requested.
    """
    parts = _safe_relpath(rel).split("/")
    fds: list[int] = []
    try:
        fd = os.open(str(repo), _O_DIR_NOFOLLOW)
        fds.append(fd)
        for name in parts[:-1]:
            try:
                nxt = os.open(name, _O_DIR_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create_missing:
                    raise StateDivergenceError(
                        f"bound path ancestor is missing: {rel}"
                    ) from None
                os.mkdir(name, dir_fd=fd)
                nxt = os.open(name, _O_DIR_NOFOLLOW, dir_fd=fd)
            except NotADirectoryError as exc:
                raise StateDivergenceError(
                    f"non-directory ancestor in bound path; refusing restore: {rel}"
                ) from exc
            except OSError as exc:
                raise StateDivergenceError(
                    f"symlink or unopenable ancestor in bound path; "
                    f"refusing restore: {rel}: {exc}"
                ) from exc
            fds.append(nxt)
            fd = nxt
        yield fd, parts[-1]
    finally:
        for handle in reversed(fds):
            try:
                os.close(handle)
            except OSError:
                pass


def _remove_bound_target_at(dir_fd: int, name: str, rel: str) -> None:
    """No-follow removal of the bound leaf: file/symlink unlink, empty-dir rmdir."""
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat_module.S_ISDIR(st.st_mode):
        probe = os.open(name, _O_DIR_NOFOLLOW, dir_fd=dir_fd)
        try:
            entries = os.listdir(probe)
        finally:
            os.close(probe)
        if entries:
            raise StateDivergenceError(
                f"directory collision contains unbound data; refusing restore: {rel}"
            )
        os.rmdir(name, dir_fd=dir_fd)
        return
    os.unlink(name, dir_fd=dir_fd)


def _write_file_at(dir_fd: int, name: str, rel: str, payload: bytes, mode: int) -> None:
    """Atomically write the bound file via dir_fd-relative temp + replace."""
    tmp_name = f".{name}.hermes-restore.tmp"
    try:
        os.unlink(tmp_name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    fd = os.open(
        tmp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=dir_fd,
    )
    replaced = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        replaced = True
    finally:
        if not replaced:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass


def _read_snapshot_object(store: Path, digest: str) -> bytes:
    """Read a snapshot object with a no-follow final open."""
    path = store / "objects" / str(digest)
    fd = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _git_dir(repo: Path) -> Path:
    raw = _git(repo, "rev-parse", "--git-dir").decode().strip()
    return ((repo / raw) if not Path(raw).is_absolute() else Path(raw)).resolve()


def _included_paths(repo: Path, ignored_paths: tuple[str, ...]) -> tuple[list[str], set[str]]:
    listed = _git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = {item.decode("utf-8", "surrogateescape") for item in listed.split(b"\0") if item}
    explicit: set[str] = set()
    for rel in ignored_paths:
        target = repo / rel
        if target.is_symlink() or target.is_file() or not target.exists():
            explicit.add(rel)
        elif target.is_dir():
            for child in target.rglob("*"):
                if child.is_symlink() or child.is_file():
                    explicit.add(child.relative_to(repo).as_posix())
    paths.update(explicit)
    return sorted(paths), explicit


def _candidate_tree(repo: Path, ignored_paths: tuple[str, ...]) -> str:
    with tempfile.TemporaryDirectory(prefix="hermes-candidate-index-") as td:
        temp_index = Path(td) / "index"
        live_index = _git_dir(repo) / "index"
        if live_index.exists():
            shutil.copyfile(live_index, temp_index)
        from tools.environments.local import build_subprocess_env
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
        env["GIT_INDEX_FILE"] = str(temp_index)
        subprocess.run(["git", "add", "-A"], cwd=repo, env=env, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ignored_paths:
            subprocess.run(["git", "add", "-f", "--", *ignored_paths], cwd=repo,
                           env=env, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        return subprocess.check_output(["git", "write-tree"], cwd=repo, env=env).decode().strip()


def _persist_snapshot(repo: Path, ignored_paths: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    paths, explicit = _included_paths(repo, ignored_paths)
    tracked_bytes = _git(repo, "ls-files", "-z")
    tracked = {item.decode("utf-8", "surrogateescape") for item in tracked_bytes.split(b"\0") if item}
    store = _git_dir(repo) / "hermes-resumable"
    objects = store / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    absent_tracked: list[str] = []
    for rel in paths:
        target = repo / rel
        if not target.exists() and not target.is_symlink():
            if rel in tracked:
                absent_tracked.append(rel)
            continue
        stat = target.lstat()
        if target.is_symlink():
            kind = "symlink"
            payload = os.readlink(target).encode("utf-8", "surrogateescape")
        elif target.is_file():
            kind = "file"
            payload = target.read_bytes()
        else:
            raise ValueError(f"unsupported checkpoint path type: {rel}")
        digest = hashlib.sha256(payload).hexdigest()
        object_path = objects / digest
        if not object_path.exists():
            fd, temporary = tempfile.mkstemp(prefix="object-", dir=str(objects))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o444)
                try:
                    os.link(temporary, object_path)
                except FileExistsError:
                    pass
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        entries.append({
            "path": rel, "kind": kind, "mode": stat.st_mode & 0o777,
            "sha256": digest, "size": len(payload), "tracked": rel in tracked,
            "explicit_ignored": rel in explicit,
        })
    manifest = {
        "schema_version": 1,
        "inclusion_policy": {
            "tracked": True, "ordinary_untracked": True,
            "explicit_ignored_paths": list(ignored_paths),
            "other_ignored": False, "special_files": False,
        },
        "entries": entries,
        "absent_tracked_paths": sorted(absent_tracked),
    }
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    snapshot_dir = store / "snapshots" / digest
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / "manifest.json"
    if path.exists() and path.read_bytes() != payload:
        raise StateDivergenceError("repository snapshot hash collision")
    if not path.exists():
        path.write_bytes(payload)
        os.chmod(path, 0o444)
    return digest, manifest


def _configured_ignored_paths() -> tuple[str, ...]:
    """Read the controller-provided explicit ignored-path inclusion list."""
    raw = os.environ.get("HERMES_RESUMABLE_IGNORED_PATHS", "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("HERMES_RESUMABLE_IGNORED_PATHS must be a JSON array") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("HERMES_RESUMABLE_IGNORED_PATHS must be a JSON string array")
    return tuple(_safe_relpath(item) for item in parsed)


def capture_repository_state(
    repository: str | Path, *,
    ignored_paths: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Capture Git identities and durable bytes under an explicit inclusion policy."""
    repo = Path(repository).resolve()
    _git(repo, "rev-parse", "--git-dir")
    selected = _configured_ignored_paths() if ignored_paths is None else ignored_paths
    normalized = tuple(_safe_relpath(item) for item in selected)
    status = _git(repo, "status", "--porcelain=v2", "-z", "--untracked-files=all")
    snapshot_sha, manifest = _persist_snapshot(repo, normalized)
    return {
        "repository": str(repo),
        "head_commit": _git(repo, "rev-parse", "HEAD").decode().strip(),
        "head_tree": _git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
        "index_tree": _git(repo, "write-tree").decode().strip(),
        "candidate_tree": _candidate_tree(repo, normalized),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "status_bytes": len(status),
        "repository_snapshot_sha256": snapshot_sha,
        "repository_snapshot_manifest_sha256": snapshot_sha,
        "repository_inclusion_policy": manifest["inclusion_policy"],
    }


def verify_repository_state(
    repository: str | Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Return live state or raise instead of continuing across drift."""
    policy = expected.get("repository_inclusion_policy") or {}
    live = capture_repository_state(
        repository, ignored_paths=policy.get("explicit_ignored_paths") or [],
    )
    keys = [
        "repository", "head_commit", "head_tree", "index_tree", "candidate_tree",
        "status_sha256", "status_bytes",
    ]
    if expected.get("repository_snapshot_sha256") is not None:
        keys.append("repository_snapshot_sha256")
    mismatches = {
        key: {"expected": expected.get(key), "actual": live.get(key)}
        for key in keys if expected.get(key) != live.get(key)
    }
    if mismatches:
        raise StateDivergenceError(
            "checkpoint repository state diverged: "
            + canonical_json_bytes(mismatches).decode("utf-8")
        )
    return live


def restore_repository_state(
    repository: str | Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore exact bound bytes; never delete post-checkpoint unbound data."""
    repo = Path(repository).resolve()
    snapshot_sha = str(expected.get("repository_snapshot_sha256") or "")
    if len(snapshot_sha) != 64:
        raise StateDivergenceError("checkpoint has no durable repository snapshot")
    if _git(repo, "rev-parse", "HEAD").decode().strip() != expected.get("head_commit"):
        raise StateDivergenceError("repository HEAD diverged; refusing restoration")
    if _git(repo, "write-tree").decode().strip() != expected.get("index_tree"):
        raise StateDivergenceError("repository index diverged; refusing restoration")
    store = _git_dir(repo) / "hermes-resumable"
    manifest_path = store / "snapshots" / snapshot_sha / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise StateDivergenceError(f"repository snapshot is unavailable: {exc}") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != snapshot_sha:
        raise StateDivergenceError("repository snapshot manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("entries") or []
    expected_paths = {str(entry["path"]) for entry in entries}
    expected_paths.update(str(item) for item in manifest.get("absent_tracked_paths") or [])
    ignored = tuple((manifest.get("inclusion_policy") or {}).get("explicit_ignored_paths") or [])
    current_paths, _ = _included_paths(repo, ignored)
    tracked = set(_git(repo, "ls-files").decode().splitlines())
    extras = sorted(path for path in current_paths if path not in expected_paths and path not in tracked)
    if extras:
        raise StateDivergenceError(
            "unbound untracked paths appeared after checkpoint; refusing destructive restore: "
            + ", ".join(extras[:10])
        )
    # Every mutation below is symlink-ancestor safe (B-003): descriptor-
    # relative with O_NOFOLLOW ancestor opens where the platform supports it,
    # otherwise an lstat rejection of every ancestor immediately before the
    # operation. A parent directory swapped for a symlink after checkpoint
    # capture must fail the restore, never redirect it outside the repo.
    if _RESTORE_DIRFD_OK:
        for rel in manifest.get("absent_tracked_paths") or []:
            rel = _safe_relpath(str(rel))
            try:
                with _parent_dirfd(repo, rel, create_missing=False) as (dfd, leaf):
                    _remove_bound_target_at(dfd, leaf, rel)
            except StateDivergenceError as exc:
                if "ancestor is missing" in str(exc):
                    continue
                raise
        for entry in entries:
            rel = _safe_relpath(str(entry["path"]))
            payload = _read_snapshot_object(store, str(entry["sha256"]))
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise StateDivergenceError(f"snapshot object digest mismatch: {rel}")
            with _parent_dirfd(repo, rel, create_missing=True) as (dfd, leaf):
                _remove_bound_target_at(dfd, leaf, rel)
                if entry["kind"] == "symlink":
                    os.symlink(
                        payload.decode("utf-8", "surrogateescape"), leaf, dir_fd=dfd,
                    )
                elif entry["kind"] == "file":
                    _write_file_at(dfd, leaf, rel, payload, int(entry["mode"]))
                else:
                    raise StateDivergenceError(f"unsupported snapshot entry kind: {rel}")
        return verify_repository_state(repo, expected)

    def remove_bound_target(target: Path, rel: str) -> None:
        if target.is_dir() and not target.is_symlink():
            try:
                next(target.iterdir())
            except StopIteration:
                target.rmdir()
                return
            raise StateDivergenceError(
                f"directory collision contains unbound data; refusing restore: {rel}"
            )
        try:
            target.unlink()
        except FileNotFoundError:
            pass

    for rel in manifest.get("absent_tracked_paths") or []:
        rel = _safe_relpath(str(rel))
        _reject_symlink_ancestors(repo, rel)
        remove_bound_target(repo / rel, rel)
    for entry in entries:
        rel = _safe_relpath(str(entry["path"]))
        target = repo / rel
        payload = _read_snapshot_object(store, str(entry["sha256"]))
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise StateDivergenceError(f"snapshot object digest mismatch: {rel}")
        _reject_symlink_ancestors(repo, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(repo, rel)
        if target.exists() or target.is_symlink():
            remove_bound_target(target, rel)
        if entry["kind"] == "symlink":
            _reject_symlink_ancestors(repo, rel)
            os.symlink(payload.decode("utf-8", "surrogateescape"), target)
        elif entry["kind"] == "file":
            _reject_symlink_ancestors(repo, rel)
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, int(entry["mode"]))
                _reject_symlink_ancestors(repo, rel)
                os.replace(temporary, target)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        else:
            raise StateDivergenceError(f"unsupported snapshot entry kind: {rel}")
    return verify_repository_state(repo, expected)


def materialize_repository_snapshot(
    expected: Mapping[str, Any], destination: str | os.PathLike[str],
) -> Path:
    """Materialize bound candidate bytes in a non-writable auditor workspace."""
    repository = expected.get("repository")
    if not isinstance(repository, str) or not repository:
        raise StateDivergenceError("checkpoint has no repository binding")
    repo = Path(repository).resolve()
    snapshot_sha = str(expected.get("repository_snapshot_sha256") or "")
    store = _git_dir(repo) / "hermes-resumable"
    manifest_path = store / "snapshots" / snapshot_sha / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise StateDivergenceError(f"repository snapshot is unavailable: {exc}") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != snapshot_sha:
        raise StateDivergenceError("repository snapshot manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    dest = Path(destination).resolve()
    if dest == repo or repo in dest.parents:
        raise StateDivergenceError("audit snapshot must be outside the candidate repository")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, mode=0o700)
    for entry in manifest["entries"]:
        rel = _safe_relpath(str(entry["path"]))
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _read_snapshot_object(store, str(entry["sha256"]))
        if len(payload) != entry["size"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise StateDivergenceError(
                f"snapshot object failed integrity check: {entry['path']}"
            )
        if entry["kind"] == "symlink":
            os.symlink(payload.decode("utf-8", "surrogateescape"), target)
            continue
        if entry["kind"] != "file":
            raise StateDivergenceError(
                f"unsupported snapshot entry kind: {entry['path']}"
            )
        target.write_bytes(payload)
        target.chmod(0o555 if int(entry["mode"]) & 0o111 else 0o444)
    # Directories are made non-writable after all descendants exist.
    directories = sorted(
        (p for p in dest.rglob("*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: len(p.parts), reverse=True,
    )
    for directory in directories:
        directory.chmod(0o555)
    dest.chmod(0o555)
    return dest


def _require_list(checkpoint: Mapping[str, Any], key: str) -> None:
    if not isinstance(checkpoint.get(key), list):
        raise ValueError(f"checkpoint {key} must be a list")


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate completeness and authority invariants without mutating input."""
    missing = sorted(REQUIRED_CHECKPOINT_FIELDS - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint missing required fields: {', '.join(missing)}")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint schema_version must be 1")
    role = checkpoint.get("execution_role")
    if role not in VALID_ROLES:
        raise ValueError(f"checkpoint execution_role must be one of {sorted(VALID_ROLES)}")
    if not isinstance(checkpoint.get("work_complete"), bool):
        raise ValueError("checkpoint work_complete must be boolean")
    for key in (
        "completed_steps", "remaining_steps", "artifacts_produced",
        "validation_performed", "validation_required", "unresolved_findings",
        "decisions", "external_actions",
    ):
        _require_list(checkpoint, key)
    if checkpoint["work_complete"] and checkpoint["remaining_steps"]:
        raise ValueError("work_complete checkpoint cannot retain remaining_steps")
    worker = checkpoint.get("worker")
    if not isinstance(worker, Mapping):
        raise ValueError("checkpoint worker must be an object")
    for field in ("identity", "model", "reasoning_effort"):
        if not str(worker.get(field) or "").strip():
            raise ValueError(f"checkpoint worker.{field} is required")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source_run_id") is None:
        raise ValueError("checkpoint provenance.source_run_id is required")
    terminal = checkpoint.get("terminal_state")
    if terminal is not None and terminal not in VALID_TERMINAL_STATES:
        raise ValueError("checkpoint terminal_state is invalid; iteration caps are not terminal")

    audit = checkpoint.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("checkpoint audit must be an object")
    for key in ("completed_criteria", "remaining_criteria"):
        if not isinstance(audit.get(key), list):
            raise ValueError(f"checkpoint audit.{key} must be a list")

    if role == "auditor":
        independent = audit.get("independence_provenance")
        emergency = provenance.get("checkpoint_quality") == "emergency-conservative"
        reviewer_identity = str(audit.get("reviewer_identity") or "")
        common_invalid = (
            not _audit_record_matches_policy(audit)
            or worker.get("identity") != reviewer_identity
            or worker.get("model") != audit.get("model")
            or worker.get("reasoning_effort") != audit.get("reasoning_effort")
            or (checkpoint.get("current_candidate_state") or {}).get("candidate_tree")
               != (checkpoint.get("bindings") or {}).get("candidate_tree")
        )
        if emergency:
            # A controller-created cap checkpoint can preserve an auditor's
            # unfinished task without claiming facts only the independent
            # desktop can attest. It cannot be CLEAN or carry completed audit
            # criteria and therefore cannot satisfy closer authority checks.
            invalid = (
                common_invalid
                or audit.get("status") != "IN_PROGRESS"
                or independent is not None
                or audit.get("signed_verdict") is not None
                or bool(audit.get("completed_criteria"))
                or checkpoint.get("work_complete") is not False
            )
        else:
            invalid = (
                common_invalid
                or not isinstance(independent, Mapping)
                or independent.get("implemented_candidate") is not False
                or independent.get("read_only") is not True
                or independent.get("fresh_invocation") is not True
                or not independent.get("producer_identity")
                or independent.get("producer_identity") == reviewer_identity
            )
        if invalid:
            raise ValueError(
                "independent auditor must match the approved ordered reviewer "
                "policy and be fresh and read-only"
            )
    if checkpoint["work_complete"]:
        if role != "closer":
            raise ValueError("work_complete is reserved for an authorized closer")
        if audit.get("status") != "CLEAN":
            raise ValueError("work_complete requires CLEAN independent audit")
    if role == "closer":
        bindings = checkpoint.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError("closer checkpoint bindings must be an object")
        authority = audit.get("authority_provenance")
        closer_status = audit.get("status")
        if closer_status not in {"CLEAN", "KILLED_BY_EVIDENCE"}:
            raise ValueError(
                "closer audit status must be CLEAN or KILLED_BY_EVIDENCE"
            )
        if closer_status == "KILLED_BY_EVIDENCE" and (
            checkpoint.get("work_complete")
            or terminal not in (None, "KILLED_BY_EVIDENCE")
        ):
            raise ValueError(
                "an evidence kill can never be work_complete and must close as "
                "KILLED_BY_EVIDENCE"
            )
        if (
            worker.get("identity") != "authorized-exact-tree-closer"
            or not _audit_record_matches_policy(audit)
            or not isinstance(authority, Mapping)
            or not authority.get("audit_task_id")
            or authority.get("audit_run_id") is None
            or not authority.get("checkpoint_sha256")
            or not bindings.get("audited_candidate_tree")
            or bindings.get("candidate_tree") != bindings.get("audited_candidate_tree")
            or (checkpoint.get("current_candidate_state") or {}).get("candidate_tree")
               != bindings.get("candidate_tree")
        ):
            raise ValueError(
                "closer requires a trusted approved-reviewer authority checkpoint "
                "and the exact audited candidate"
            )


def build_emergency_checkpoint(
    *, task: Any, run_id: int, agent: Any, summary: str | None,
    previous: Mapping[str, Any] | None = None,
    yield_reason: str = "ITERATION_CAP_REACHED",
) -> dict[str, Any]:
    """Create a conservative full checkpoint when no rich heartbeat exists.

    Unknown progress is never guessed: prior durable fields are preserved and
    the continuation is instructed to reconcile artifacts before proceeding.
    """
    from datetime import datetime, timezone

    prior = dict(previous or {})
    repo_state: dict[str, Any]
    workspace_path = getattr(task, "workspace_path", None)
    try:
        repo_state = capture_repository_state(str(workspace_path or ""))
    except Exception as exc:
        repo_state = {
            "repository": str(workspace_path or ""),
            "capture_error": str(exc),
            "head_commit": None, "head_tree": None, "index_tree": None,
            "candidate_tree": None, "status_sha256": None, "status_bytes": None,
        }
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    worker_identity = str(
        os.environ.get("HERMES_WORKER_IDENTITY")
        or os.environ.get("HERMES_PROFILE")
        or getattr(task, "assignee", None)
        or "kanban-worker"
    )
    role = str(os.environ.get("HERMES_EXECUTION_ROLE") or prior.get("execution_role") or "implementer")
    if role not in VALID_ROLES:
        role = "implementer"
    model = str(
        getattr(task, "model_override", None)
        or getattr(agent, "model", None)
        or os.environ.get("HERMES_MODEL")
        or "unknown"
    )
    effort = str(
        getattr(task, "reasoning_effort", None)
        or getattr(agent, "reasoning_effort", None)
        or os.environ.get("HERMES_REASONING_EFFORT")
        or "unknown"
    )
    previous_bindings: Mapping[str, Any] = dict(prior.get("bindings") or {}) if isinstance(prior.get("bindings"), Mapping) else {}
    previous_audit: Mapping[str, Any] = dict(prior.get("audit") or {}) if isinstance(prior.get("audit"), Mapping) else {}
    previous_timestamps: Mapping[str, Any] = dict(prior.get("timestamps") or {}) if isinstance(prior.get("timestamps"), Mapping) else {}
    completed = list(prior.get("completed_steps") or [])
    remaining = list(prior.get("remaining_steps") or [])
    if not remaining:
        remaining = [
            "Reconcile durable repository artifacts against this checkpoint",
            "Continue the authorized objective from the nearest verified phase",
            "Complete required validation, independent audit, and close steps",
        ]
    audit_state = {
        "status": previous_audit.get("status") or "NOT_STARTED",
        "reviewer_identity": previous_audit.get("reviewer_identity"),
        "model": previous_audit.get("model"),
        "reasoning_effort": previous_audit.get("reasoning_effort"),
        "independence_provenance": previous_audit.get("independence_provenance"),
        "completed_criteria": list(previous_audit.get("completed_criteria") or []),
        "remaining_criteria": list(previous_audit.get("remaining_criteria") or []),
    }
    if role == "auditor" and not prior:
        audit_state = {
            "status": "IN_PROGRESS", "reviewer_identity": AUDITOR_IDENTITY,
            "model": AUDITOR_MODEL, "reasoning_effort": AUDITOR_EFFORT,
            # Do not manufacture independence, read-only, freshness, or producer
            # identity from the worker environment. This emergency checkpoint
            # carries progress only; a rich auditor checkpoint plus a verified
            # desktop signature is still required for authority.
            "independence_provenance": None,
            "completed_criteria": [],
            "remaining_criteria": ["reconcile full independent audit criteria"],
        }
    return {
        "schema_version": 1,
        "task_id": task.id,
        "task_version": str(getattr(task, "created_at", "1")),
        "task_objective": str(task.title) + (("\n" + task.body) if getattr(task, "body", None) else ""),
        "authorized_scope": prior.get("authorized_scope") or {
            "workspace_kind": getattr(task, "workspace_kind", None),
            "repository": workspace_path,
        },
        "execution_role": role,
        "worker": {"identity": worker_identity, "model": model, "reasoning_effort": effort},
        "current_phase": prior.get("current_phase") or getattr(task, "current_step_key", None) or "reconciliation",
        "phase_status": "IN_PROGRESS",
        "work_complete": False,
        "base_commit": prior.get("base_commit") or repo_state.get("head_commit"),
        "base_tree": prior.get("base_tree") or repo_state.get("head_tree"),
        "current_candidate_state": repo_state,
        "bindings": {
            "candidate_tree": repo_state.get("candidate_tree"),
            "dossier_sha256": previous_bindings.get("dossier_sha256"),
            "contract_sha256": previous_bindings.get("contract_sha256"),
            "evidence_sha256": list(previous_bindings.get("evidence_sha256") or []),
            **({"audited_candidate_tree": previous_bindings.get("audited_candidate_tree")}
               if previous_bindings.get("audited_candidate_tree") else {}),
        },
        "completed_steps": completed,
        "remaining_steps": remaining,
        "artifacts_produced": list(prior.get("artifacts_produced") or []),
        "validation_performed": list(prior.get("validation_performed") or []),
        "validation_required": list(prior.get("validation_required") or ["Re-run governing validation not proven complete"]),
        "audit": audit_state,
        "unresolved_findings": list(prior.get("unresolved_findings") or []),
        "decisions": list(prior.get("decisions") or []),
        "external_actions": list(prior.get("external_actions") or []),
        "continuation_instruction": (
            "Restore this checkpoint in a clean invocation. Verify the bound "
            "repository state before new work; inspect produced artifacts to "
            "reconcile any progress not present in completed_steps. Do not repeat "
            "external_actions. Resume the recorded role and phase."
        ),
        "retry_count": int(prior.get("retry_count") or 0),
        "timestamps": {"created_at_utc": previous_timestamps.get("created_at_utc", now), "updated_at_utc": now},
        "provenance": {
            "source_run_id": run_id,
            "yield_reason": str(yield_reason),
            "summary_sha256": hashlib.sha256((summary or "").encode("utf-8")).hexdigest(),
            "checkpoint_quality": "emergency-conservative",
        },
    }


def assert_continuation_needed(checkpoint: Mapping[str, Any]) -> None:
    validate_checkpoint(checkpoint)
    terminal = checkpoint.get("terminal_state")
    if terminal in (VALID_TERMINAL_STATES - {"FAILED_RECOVERABLE"}):
        raise ValueError("terminal checkpoint must not enqueue a continuation")
    if checkpoint["work_complete"]:
        raise ValueError("work_complete checkpoint must not enqueue a continuation")
    if not checkpoint["remaining_steps"] and not checkpoint["validation_required"]:
        raise ValueError("unfinished checkpoint must identify remaining work")
