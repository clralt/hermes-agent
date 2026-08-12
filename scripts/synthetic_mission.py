#!/usr/bin/env python3
"""Accelerated synthetic mission for disposable-worker baton verification.

This harness exercises the production-shaped contract without making model/API
calls: a controller owns durable JSON state, launches one bounded worker at a
time, consumes a structured result, and dies/restarts between slices. Workers
are disposable subprocesses. Crashes and audit failures are injected at named
logical steps, then the controller retries from the last durable checkpoint.

It is deliberately local-only and side-effect free outside ``--root``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXIT_CONTROLLER_RESTART = 75
EXIT_CONTROLLER_CRASH = 76


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def load_state(root: Path) -> dict[str, Any]:
    path = root / "controller-state.json"
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "mission_id": "synthetic-6h-baton",
            "status": "READY",
            "next_step": 0,
            "completed_steps": [],
            "pending": None,
            "events": [],
            "worker_turnovers": 0,
            "worker_crashes": 0,
            "audit_failures": 0,
            "controller_restarts": 0,
            "controller_crashes": 0,
            "controller_crash_injected_steps": [],
            "checkpoints": 0,
            "checkpoint_ids": [],
            "logical_duration_seconds": 0,
            "human_interventions": 0,
            "active_worker": None,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    # Additive state migration keeps a controller restart compatible with a
    # checkpoint written by an earlier harness revision.
    state.setdefault("controller_crashes", 0)
    state.setdefault("controller_crash_injected_steps", [])
    state.setdefault("checkpoint_ids", [])
    state.setdefault("active_worker", None)
    state.setdefault("pending", None)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported mission state schema")
    if state.get("active_worker") is not None:
        raise RuntimeError("controller found an active worker after restart")
    if state.get("pending") is not None:
        pending = state["pending"]
        if pending.get("step") != state.get("next_step"):
            raise RuntimeError("pending baton does not match next_step")
    return state


def record(state: dict[str, Any], kind: str, **data: Any) -> None:
    event = {"seq": len(state["events"]) + 1, "kind": kind, **data}
    state["events"].append(event)


def save_checkpoint(root: Path, state: dict[str, Any], *, reason: str) -> None:
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "mission_id": state["mission_id"],
        "step": state["next_step"],
        "completed_steps": list(state["completed_steps"]),
        "pending": state["pending"],
        "reason": reason,
        "controller_restart_safe": True,
    }
    state["checkpoints"] += 1
    checkpoint_id = digest(checkpoint)
    state["checkpoint_ids"].append(checkpoint_id)
    record(state, "checkpoint_written", step=state["next_step"], reason=reason, checkpoint_id=checkpoint_id, sha256=checkpoint_id)
    atomic_json(root / "checkpoints" / f"step-{state['next_step']:04d}-{checkpoint_id[:12]}.json", checkpoint)
    atomic_json(root / "controller-state.json", state)


def worker(root: Path, step: int, worker_id: str, *, crash: bool, audit_fail: bool) -> int:
    result = {
        "schema_version": SCHEMA_VERSION,
        "worker_id": worker_id,
        "step": step,
        "bounded_work": {"operation": "advance-synthetic-unit", "unit": step},
        "artifact": {"path": f"artifacts/unit-{step:04d}.json", "value": step * 2 + 1},
        "audit": {"status": "FAIL" if audit_fail else "PASS", "criteria": ["structured-result", "single-baton"]},
        "finished_at": time.time(),
    }
    result_path = root / "results" / f"worker-{worker_id}.json"
    if crash:
        # Persist no success result. The controller must observe the non-zero
        # process outcome and launch a fresh worker for the same step.
        os.kill(os.getpid(), signal.SIGKILL)
    atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def launch_worker(root: Path, step: int, *, crash: bool, audit_fail: bool) -> dict[str, Any]:
    worker_id = f"w-{step:04d}-{uuid.uuid4().hex[:8]}"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--root", str(root), "--step", str(step), "--worker-id", worker_id]
    if crash:
        command.append("--crash")
    if audit_fail:
        command.append("--audit-fail")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"worker_crash:{worker_id}:exit={completed.returncode}")
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"worker_result_invalid:{worker_id}") from exc
    validate_worker_result(result, worker_id=worker_id, step=step)
    return result


def validate_worker_result(result: Any, *, worker_id: str, step: int) -> None:
    """Reject malformed, misbound, or path-escaping worker batons."""
    if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"worker_result_invalid:{worker_id}:schema")
    if result.get("worker_id") != worker_id or result.get("step") != step:
        raise RuntimeError(f"worker_result_binding_mismatch:{worker_id}")
    audit = result.get("audit")
    if not isinstance(audit, dict) or audit.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError(f"worker_result_invalid:{worker_id}:audit")
    artifact = result.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"worker_result_invalid:{worker_id}:artifact")
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.startswith("artifacts/") or ".." in Path(artifact_path).parts:
        raise RuntimeError(f"worker_result_invalid:{worker_id}:artifact-path")
    if not isinstance(artifact.get("value"), int):
        raise RuntimeError(f"worker_result_invalid:{worker_id}:artifact")


def controller(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = load_state(root)
    state["controller_restarts"] += 1
    state["status"] = "RUNNING"
    if args.controller_crash_step is not None and state["next_step"] == args.controller_crash_step and state["next_step"] not in state["controller_crash_injected_steps"]:
        state["controller_crash_injected_steps"].append(state["next_step"])
        state["controller_crashes"] += 1
        state["active_worker"] = None
        record(state, "controller_crashed", step=state["next_step"])
        save_checkpoint(root, state, reason="controller_crash")
        return EXIT_CONTROLLER_CRASH
    atomic_json(root / "controller-state.json", state)
    crash_steps = set(args.crash_steps)
    audit_fail_steps = set(args.audit_fail_steps)
    slice_end = min(args.steps, state["next_step"] + args.controller_slice)
    while state["next_step"] < slice_end:
        step = state["next_step"]
        state["active_worker"] = {"step": step, "controller_pid": os.getpid()}
        state["pending"] = {"step": step, "attempt": len([e for e in state["events"] if e.get("kind") == "worker_started" and e.get("step") == step]) + 1}
        state["worker_turnovers"] += 1
        record(state, "worker_started", step=step, attempt=state["pending"]["attempt"])
        atomic_json(root / "controller-state.json", state)
        try:
            result = launch_worker(root, step, crash=step in crash_steps and state["pending"]["attempt"] == 1, audit_fail=step in audit_fail_steps and state["pending"]["attempt"] == 1)
        except RuntimeError as exc:
            if str(exc).startswith("worker_crash:"):
                state["worker_crashes"] += 1
                state["active_worker"] = None
                record(state, "worker_crashed", step=step)
                save_checkpoint(root, state, reason="worker_crash")
                continue
            state["status"] = "FAILED"
            state["active_worker"] = None
            atomic_json(root / "controller-state.json", state)
            raise
        state["active_worker"] = None
        if result["audit"]["status"] == "FAIL":
            state["audit_failures"] += 1
            record(state, "audit_failed", step=step, worker_id=result["worker_id"])
            state["pending"] = {"step": step, "retry_reason": "audit_failure"}
            save_checkpoint(root, state, reason="audit_failure")
            continue
        artifact = root / result["artifact"]["path"]
        atomic_json(artifact, {"step": step, "value": result["artifact"]["value"], "worker_id": result["worker_id"]})
        state["completed_steps"].append(step)
        state["next_step"] += 1
        state["pending"] = None
        state["logical_duration_seconds"] = int(state["next_step"] * (args.logical_hours * 3600 / args.steps))
        record(state, "worker_result_accepted", step=step, worker_id=result["worker_id"])
        save_checkpoint(root, state, reason="accepted_result")
    if state["next_step"] < args.steps:
        state["status"] = "CONTINUATION_REQUIRED"
        state["active_worker"] = None
        atomic_json(root / "controller-state.json", state)
        return EXIT_CONTROLLER_RESTART
    state["status"] = "COMPLETED"
    state["active_worker"] = None
    state["pending"] = None
    state["zero_human_intervention"] = state["human_interventions"] == 0
    atomic_json(root / "controller-state.json", state)
    summary = {
        "status": state["status"],
        "mission_id": state["mission_id"],
        "completed_steps": len(state["completed_steps"]),
        "worker_turnovers": state["worker_turnovers"],
        "worker_crashes": state["worker_crashes"],
        "audit_failures": state["audit_failures"],
        "controller_restarts": state["controller_restarts"],
        "controller_crashes": state["controller_crashes"],
        "checkpoints": state["checkpoints"],
        "logical_duration_seconds": state["logical_duration_seconds"],
        "max_concurrent_workers": 1,
        "human_interventions": state["human_interventions"],
        "zero_human_intervention": state["zero_human_intervention"],
        "event_log_sha256": digest(state["events"]),
    }
    atomic_json(root / "mission-summary.json", summary)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def supervise(args: argparse.Namespace) -> int:
    """Run each controller slice as a fresh process.

    The supervisor is intentionally thinner than the controller: it owns no
    progress state and only reacts to the durable controller exit protocol.
    A real controller process therefore dies on every slice (and on injected
    crash), proving that the baton lives outside both worker and controller.
    """
    command = [sys.executable, str(Path(__file__).resolve()), "--controller"]
    command.extend(item for item in sys.argv[1:] if item != "--supervise")
    while True:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode in (EXIT_CONTROLLER_RESTART, EXIT_CONTROLLER_CRASH):
            continue
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="hermes-synthetic-self-test-") as raw:
        root = Path(raw)
        state = load_state(root)
        state["next_step"] = 4
        state["pending"] = {"step": 4}
        atomic_json(root / "controller-state.json", state)
        restored = load_state(root)
        if restored["pending"]["step"] != restored["next_step"]:
            return 1
    print("self_test=PASS")
    return 0


def parse_steps(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--controller", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", default=".runtime/synthetic-mission")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--controller-slice", type=int, default=4)
    parser.add_argument("--crash-steps", type=parse_steps, default=[])
    parser.add_argument("--audit-fail-steps", type=parse_steps, default=[])
    parser.add_argument("--controller-crash-step", type=int)
    parser.add_argument("--logical-hours", type=float, default=6.0)
    parser.add_argument("--step", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--audit-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.worker:
        if args.step is None or not args.worker_id:
            parser.error("--worker requires --step and --worker-id")
        return worker(Path(args.root).resolve(), args.step, args.worker_id, crash=args.crash, audit_fail=args.audit_fail)
    if args.steps <= 0 or args.controller_slice <= 0:
        parser.error("--steps and --controller-slice must be positive")
    if args.supervise:
        return supervise(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
