"""CPU-only runtime contracts with real external commands and Git snapshots."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omnirsi.contracts import RunSpec
from omnirsi.execution import repository_identity, run_experiment
from omnirsi.reporting import write_report


def benchmark(value: object) -> tuple[str, ...]:
    code = "import json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
    return (sys.executable, "-c", code, "{result}", json.dumps({"metrics": {"latency_ms": value}}))


def process_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE, no termination rights
        if not handle:
            if ctypes.get_last_error() == 87:  # process has exited
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return kernel.WaitForSingleObject(handle, 1000) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
        return False
    return True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "source.txt").write_text("initial source\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", "source.txt")
        self.git("-c", "user.name=OmniRSI test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-s", "-m", "fixture")
        self.spec = RunSpec(repo=str(self.repo), baseline_command=benchmark(100),
                            candidate_command=benchmark(80), repetitions=2,
                            validation_command=(sys.executable, "-c", "print('caller quality checks passed')"),
                            output_dir=str(self.root / "runs"), max_minutes=1)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_spec(self, **changes):
        path = run_experiment(replace(self.spec, **changes), {"backend": "cpu", "test": True})
        return path, json.loads((path / "evaluation.json").read_text(encoding="utf-8"))

    def test_positive_gain_with_explicit_quality_passes_and_alternates_order(self):
        path, evaluation = self.run_spec()
        self.assertEqual(evaluation["verdict"], "PASS")
        self.assertAlmostEqual(evaluation["improvement_pct"], 20)
        events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["side"] for event in events if event["phase"] == "measurement"],
                         ["baseline", "candidate", "candidate", "baseline"])
        trial = json.loads((path / "baseline/trial-001/execution.json").read_text(encoding="utf-8"))
        self.assertGreater(trial["pid"], 0)
        self.assertIn(str(path / "baseline/trial-001/result.json"), trial["argv"])
        self.assertEqual(json.loads((path / "status.json").read_text())["state"], "completed")
        self.assertTrue((path / "report.html").is_file())

    def test_missing_quality_evidence_is_inconclusive(self):
        _, evaluation = self.run_spec(validation_command=None)
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertTrue(any("quality" in reason for reason in evaluation["reasons"]))

    def test_quality_failure_and_benchmark_failure_are_fail(self):
        failure = (sys.executable, "-c", "import sys; print('failed evidence'); sys.exit(7)")
        path, evaluation = self.run_spec(validation_command=failure)
        self.assertEqual(evaluation["verdict"], "FAIL")
        self.assertIn("failed evidence", (path / "validation/stdout.log").read_text())
        path, evaluation = self.run_spec(baseline_command=failure)
        self.assertEqual(evaluation["verdict"], "FAIL")
        self.assertFalse((path / "candidate").exists())
        self.assertEqual(json.loads((path / "baseline/trial-001/execution.json").read_text())["returncode"], 7)

    def test_missing_malformed_nonfinite_and_nonpositive_metrics_are_inconclusive(self):
        commands = [
            (sys.executable, "-c", "pass"),
            (sys.executable, "-c", "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('{broken')", "{result}"),
            benchmark("80"), benchmark(True), benchmark(float("nan")), benchmark(float("inf")),
            benchmark(0), benchmark(-1),
        ]
        for command in commands:
            with self.subTest(command=command):
                _, evaluation = self.run_spec(candidate_command=command, repetitions=1)
                self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")

    def test_result_environment_variable_is_supported(self):
        code = "import json, os, pathlib; pathlib.Path(os.environ['OMNIRSI_RESULT_PATH']).write_text(json.dumps({'metrics':{'latency_ms':80}}))"
        _, evaluation = self.run_spec(candidate_command=(sys.executable, "-c", code), repetitions=1)
        self.assertEqual(evaluation["verdict"], "PASS")

    def test_direction_threshold_and_equal_values(self):
        _, evaluation = self.run_spec(min_improvement_pct=21)
        self.assertEqual(evaluation["verdict"], "FAIL")
        _, evaluation = self.run_spec(candidate_command=benchmark(100))
        self.assertEqual(evaluation["verdict"], "FAIL")
        _, evaluation = self.run_spec(direction="maximize", candidate_command=benchmark(120))
        self.assertEqual(evaluation["verdict"], "PASS")
        _, evaluation = self.run_spec(direction="maximize", candidate_command=benchmark(80))
        self.assertEqual(evaluation["verdict"], "FAIL")

    def test_source_change_is_inconclusive_and_existing_dirty_source_is_recorded(self):
        (self.repo / "source.txt").write_text("pre-existing local change\n", encoding="utf-8")
        identity = repository_identity(self.repo)
        self.assertTrue(identity["available"])
        self.assertTrue(identity["dirty"])
        _, evaluation = self.run_spec()
        self.assertEqual(evaluation["verdict"], "PASS")
        code = ("import json, pathlib, sys; pathlib.Path('source.txt').write_text('unexpected change'); "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({'metrics': {'latency_ms':80}}))")
        path, evaluation = self.run_spec(candidate_command=(sys.executable, "-c", code, "{result}"))
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertIn("source changed", evaluation["source_integrity"]["reason"])
        self.assertTrue((path / "repositories.final.json").is_file())

    def test_source_change_restored_later_still_invalidates_comparison(self):
        changed = ("import json, pathlib, sys; pathlib.Path('source.txt').write_text('temporary drift'); "
                   "pathlib.Path(sys.argv[1]).write_text(json.dumps({'metrics': {'latency_ms':100}}))")
        restored = ("import json, pathlib, sys; pathlib.Path('source.txt').write_text('initial source\\n'); "
                    "pathlib.Path(sys.argv[1]).write_text(json.dumps({'metrics': {'latency_ms':80}}))")
        path, evaluation = self.run_spec(baseline_command=(sys.executable, "-c", changed, "{result}"),
                                         candidate_command=(sys.executable, "-c", restored, "{result}"), repetitions=1)
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertEqual((self.repo / "source.txt").read_text(), "initial source\n")
        checks = json.loads((path / "source_checks.json").read_text())
        self.assertNotEqual(checks[0]["before"]["snapshot_hash"], checks[0]["after"]["snapshot_hash"])

    def test_derived_metric_overflow_preserves_inconclusive_artifacts(self):
        path, evaluation = self.run_spec(baseline_command=benchmark(1e-308), candidate_command=benchmark(1e308),
                                         repetitions=1)
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertTrue((path / "report.html").is_file())

    def test_unknown_source_identity_is_inconclusive(self):
        plain = self.root / "plain"
        plain.mkdir()
        self.assertFalse(repository_identity(plain)["available"])
        _, evaluation = self.run_spec(candidate_repo=str(plain), repetitions=1)
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")

    def test_outputs_inside_repo_do_not_create_false_source_changes(self):
        _, evaluation = self.run_spec(output_dir=str(self.repo / "runs"))
        self.assertEqual(evaluation["verdict"], "PASS")

    def test_unique_runs_do_not_reuse_previous_metrics(self):
        first, _ = self.run_spec(repetitions=1)
        second, evaluation = self.run_spec(candidate_command=(sys.executable, "-c", "pass"), repetitions=1)
        self.assertNotEqual(first, second)
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertFalse((second / "candidate/trial-001/result.json").exists())

    def test_timeout_kills_child_tree_and_preserves_evidence(self):
        child = "import time; time.sleep(30)"
        parent = "import subprocess, sys, time; p=subprocess.Popen([sys.executable, '-c', sys.argv[1]]); print(p.pid, flush=True); time.sleep(30)"
        path, evaluation = self.run_spec(baseline_command=(sys.executable, "-c", parent, child),
                                         max_minutes=0.1, repetitions=1)
        self.assertEqual(evaluation["verdict"], "FAIL")
        self.assertEqual(json.loads((path / "status.json").read_text())["state"], "timeout")
        child_pid = int((path / "baseline/trial-001/stdout.log").read_text().strip())
        self.assertFalse(process_running(child_pid), "benchmark descendant survived the timeout")

    def test_deadline_also_applies_to_validation(self):
        path, evaluation = self.run_spec(validation_command=(sys.executable, "-c", "import time; time.sleep(30)"),
                                         repetitions=1, max_minutes=0.25)
        self.assertEqual(evaluation["verdict"], "FAIL")
        self.assertEqual(json.loads((path / "validation/execution.json").read_text())["status"], "timeout")

    def test_successful_command_also_cleans_up_background_descendants(self):
        child = "import time; time.sleep(30)"
        parent = ("import json, pathlib, subprocess, sys; "
                  "p=subprocess.Popen([sys.executable, '-c', sys.argv[1]]); print(p.pid, flush=True); "
                  "pathlib.Path(sys.argv[2]).write_text(json.dumps({'metrics': {'latency_ms':80}}))")
        path, evaluation = self.run_spec(candidate_command=(sys.executable, "-c", parent, child, "{result}"),
                                       repetitions=1)
        self.assertEqual(evaluation["verdict"], "PASS")
        child_pid = int((path / "candidate/trial-001/stdout.log").read_text().strip())
        self.assertFalse(process_running(child_pid), "completed benchmark left a background descendant")

    def test_keyboard_interrupt_preserves_cancelled_run(self):
        with patch("omnirsi.execution._execute", side_effect=KeyboardInterrupt):
            path, evaluation = self.run_spec()
        self.assertEqual(json.loads((path / "status.json").read_text())["state"], "cancelled")
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE")
        self.assertTrue((path / "report.html").exists())

    def test_report_escapes_html_and_is_regenerable(self):
        path, _ = self.run_spec(scenario="<script>alert('x')</script>", repetitions=1)
        report = (path / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert", report)
        self.assertIn("&lt;script&gt;", report)
        (path / "report.html").unlink()
        self.assertEqual(write_report(path), path / "report.html")

    def test_contract_rejects_invalid_inputs_before_allocating_run(self):
        for changes in ({"repetitions": 0}, {"direction": "unknown"}, {"max_minutes": float("inf")},
                        {"min_improvement_pct": -1}, {"baseline_command": "echo unsafe"},
                        {"metric_key": "metrics..latency"}, {"output_dir": str(self.repo)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_experiment(replace(self.spec, **changes), {})


if __name__ == "__main__":
    unittest.main()
