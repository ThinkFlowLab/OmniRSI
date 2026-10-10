import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = (
    Path(__file__).resolve().parents[1] / "examples/qwen21_turbo_validation_adapter.py"
)
spec = importlib.util.spec_from_file_location("qwen21_validation_adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ValidationAdapterTests(unittest.TestCase):
    def test_nested_runner_result_resolves_original_run_without_changing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.lock.json").write_text(json.dumps({"repetitions": 5}))
            result = root / "validation/result.json"
            env = adapter.validation_environment(
                result, {"OMNIRSI_RESULT_PATH": str(result), "CONTROL": "frozen"}
            )
            self.assertEqual(Path(env["OMNIRSI_RESULT_PATH"]).parent, root)
            self.assertEqual(env["CONTROL"], "frozen")
            self.assertFalse((root / "v4-root-adapter.json").exists())

    def test_wrong_location_and_incomplete_cohort_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                adapter.validation_environment(root / "result.json", {})
            (root / "run.lock.json").write_text(json.dumps({"repetitions": 4}))
            with self.assertRaises(ValueError):
                adapter.validation_environment(root / "validation/result.json", {})
