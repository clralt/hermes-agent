from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "synthetic_mission.py"


class SyntheticMissionTests(unittest.TestCase):
    def run_mission(self, *, steps: int = 32, controller_slice: int = 4) -> tuple[dict, Path]:
        with tempfile.TemporaryDirectory(prefix="hermes-synthetic-mission-") as raw:
            root = Path(raw)
            command = [
                sys.executable,
                str(SCRIPT),
                "--supervise",
                "--root",
                str(root),
                "--steps",
                str(steps),
                "--controller-slice",
                str(controller_slice),
                "--crash-steps",
                "2,7",
                "--audit-fail-steps",
                "4,9",
                "--controller-crash-step",
                "16",
                "--logical-hours",
                "6",
                "--json",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["completed_steps"], steps)
            self.assertGreaterEqual(summary["worker_turnovers"], steps + 4)
            self.assertGreaterEqual(summary["worker_crashes"], 2)
            self.assertGreaterEqual(summary["audit_failures"], 2)
            self.assertGreaterEqual(summary["controller_restarts"], 1)
            self.assertGreaterEqual(summary["controller_crashes"], 1)
            self.assertEqual(summary["max_concurrent_workers"], 1)
            self.assertGreaterEqual(summary["checkpoints"], steps)
            self.assertGreaterEqual(summary["logical_duration_seconds"], 6 * 3600)
            self.assertTrue(summary["zero_human_intervention"])
            return summary, root

    def test_disposable_workers_pass_baton_after_crashes_and_audit_failures(self) -> None:
        self.run_mission()

    def test_mission_is_idempotently_resumable_from_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hermes-synthetic-resume-") as raw:
            root = Path(raw)
            first = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--controller",
                    "--root",
                    str(root),
                    "--steps",
                    "8",
                    "--controller-slice",
                    "2",
                    "--crash-steps",
                    "3",
                    "--audit-fail-steps",
                    "5",
                    "--controller-crash-step",
                    "4",
                    "--logical-hours",
                    "6",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(first.returncode, 75, first.stderr)
            second = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--supervise",
                    "--root",
                    str(root),
                    "--steps",
                    "8",
                    "--controller-slice",
                    "2",
                    "--crash-steps",
                    "3",
                    "--audit-fail-steps",
                    "5",
                    "--controller-crash-step",
                    "4",
                    "--logical-hours",
                    "6",
                    "--json",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            summary = json.loads(second.stdout)
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["completed_steps"], 8)
            self.assertGreaterEqual(summary["controller_restarts"], 2)
            self.assertGreaterEqual(summary["controller_crashes"], 1)

    def test_invalid_worker_baton_paths_fail_closed(self) -> None:
        script = """
import importlib.util
spec = importlib.util.spec_from_file_location('mission', r'''%s''')
mission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mission)
try:
    mission.validate_worker_result({"schema_version": 1, "worker_id": "w", "step": 1, "audit": {"status": "PASS"}, "artifact": {"path": "../escape", "value": 1}}, worker_id="w", step=1)
except RuntimeError as exc:
    assert "artifact-path" in str(exc)
else:
    raise SystemExit("path escape accepted")
""" % SCRIPT
        completed = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_result_or_duplicate_baton_fails_closed(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--self-test"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("self_test=PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
