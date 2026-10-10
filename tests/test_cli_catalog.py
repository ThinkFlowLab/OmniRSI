"""CLI contracts, provenance and scope boundaries without model dependencies."""

import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from omnirsi.arguments import make_spec, parser
from omnirsi.catalog import get_knowledge, knowledge_records, repository_pack, search_knowledge
from omnirsi.cli import main

PROJECT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_source_grounded_imports_and_backend_boundaries(self):
        records = knowledge_records()
        self.assertGreaterEqual(len(records), 10)
        self.assertEqual(len(records), len({item["id"] for item in records}))
        for record in records:
            self.assertEqual(record["status"], "imported")
            self.assertFalse(record["verification"]["locally_verified"])
            for source in record["sources"]:
                self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
                self.assertTrue(source["url"].startswith("https://github.com/vllm-project/vllm-omni/"))
        rocm = search_knowledge(repo="vllm_omni", backend="rocm", limit=50)
        self.assertTrue(rocm)
        self.assertTrue(all(item["backend"] == "common" for item in rocm))
        self.assertEqual(search_knowledge(repo="vllm_rlt", limit=50), [])

    def test_scenario_alias_and_general_methods(self):
        direct = search_knowledge(repo="vllm_omni", backend="cuda", scenario="diffusion.video_generation", limit=50)
        alias = search_knowledge(repo="vllm_omni", backend="cuda", scenario="diffusion.t2v", limit=50)
        self.assertEqual([item["id"] for item in direct], [item["id"] for item in alias])
        self.assertGreaterEqual(len([item for item in direct if item["backend"] == "common"]), 2)
        self.assertTrue(all(item["backend"] in ("cuda", "common") for item in direct))

    def test_unknown_and_invalid_search_parameters(self):
        with self.assertRaises(ValueError):
            get_knowledge("not-a-record")
        with self.assertRaises(ValueError):
            search_knowledge(limit=0)
        for name in ("vllm", "vllm_omni", "afd_plugin", "vllm_rlt"):
            pack = repository_pack(name)
            self.assertEqual(pack["execution_support"], "caller_supplied_external_commands")
            self.assertTrue(all(value == "unverified" for value in pack["backend_compatibility"].values()))

    def test_context_retains_conditions_and_source_verification(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["context", "MiniMax-H3", "--repo-type", "vllm_omni", "--backend", "ascend", "--limit", "2"])
        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn('"applicability"', text)
        self.assertIn('"claims"', text)
        self.assertIn('"locally_verified": false', text)
        self.assertIn("not tool instructions", text)
        self.assertNotIn("vllm_omni.cuda", text)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omnirsi-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        subprocess.run(["git", "init", str(self.repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "core.hooksPath=" + str(self.root / "no-hooks"),
                        "-c", "user.name=OmniRSI Test", "-c", "user.email=test@example.invalid",
                        "commit", "-s", "--allow-empty", "-m", "fixture"], check=True, capture_output=True)
        self.output_dir = self.root / "evidence"
        benchmark = PROJECT / "examples/synthetic_benchmark.py"
        validation = PROJECT / "examples/synthetic_validation.py"
        self.arguments = ["run", "--repo", str(self.repo), "--repo-type", "vllm_omni", "--backend", "cpu",
                          "--baseline-command", shlex.join([sys.executable, str(benchmark), "--value", "100", "--output", "{result}"]),
                          "--candidate-command", shlex.join([sys.executable, str(benchmark), "--value", "80", "--output", "{result}"]),
                          "--validation-command", shlex.join([sys.executable, str(validation)]),
                          "--repetitions", "1", "--max-minutes", "1", "--output-dir", str(self.output_dir)]

    def invoke(self, arguments):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(arguments)
        return code, json.loads(stream.getvalue())

    def test_cli_experiment_and_report(self):
        code, result = self.invoke(self.arguments)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"]["verdict"], "PASS")
        directory = Path(result["run_dir"])
        self.assertTrue((directory / "args.json").is_file())
        evaluation = json.loads((directory / "evaluation.json").read_text())
        self.assertEqual(evaluation["improvement_pct"], 20.0)
        self.assertEqual(evaluation["statistical_significance"], "not established")
        code, status = self.invoke(["status", "--run-id", result["run_id"], "--output-dir", str(self.output_dir)])
        self.assertEqual(status["state"], "completed")
        code, report = self.invoke(["report", "--run-id", result["run_id"], "--output-dir", str(self.output_dir)])
        self.assertTrue(Path(report["report"]).is_file())

    def test_dry_run_is_not_execution(self):
        code, result = self.invoke(self.arguments + ["--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(result["execution"], "not started")
        self.assertFalse(self.output_dir.exists())

    def test_missing_validation_is_inconclusive(self):
        arguments = self.arguments.copy()
        start = arguments.index("--validation-command")
        del arguments[start:start + 2]
        code, result = self.invoke(arguments)
        self.assertEqual(code, 2)
        self.assertEqual(result["status"]["verdict"], "INCONCLUSIVE")

    def test_input_errors_are_not_silent(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(self.arguments + ["--repetitions", "0"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(self.arguments + ["--max-minutes", "1e308"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(self.arguments + ["--device-ids", "0,0"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["status", "--run-id", "../outside", "--output-dir", str(self.output_dir)])

    def test_argv_metacharacters_are_literal(self):
        arguments = self.arguments.copy()
        start = arguments.index("--baseline-command") + 1
        arguments[start] = "tool 'a; b' '{result}'"
        spec = make_spec(parser().parse_args(arguments))
        self.assertEqual(spec.baseline_command, ("tool", "a; b", "{result}"))


if __name__ == "__main__":
    unittest.main()
