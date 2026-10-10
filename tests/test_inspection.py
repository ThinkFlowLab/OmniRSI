"""Inspection facts must not become accelerator compatibility claims."""

import json
import unittest
from unittest.mock import patch

from omnirsi.inspection import inspect_environment


class InspectionTests(unittest.TestCase):
    def inspect_cuda(self, rows, ids=("0",), model=None):
        def command(argv):
            if "-c" in argv:
                stdout = json.dumps({"python": "test", "packages": {"torch": "test-version"}})
            elif "topo" in argv:
                stdout = "test topology"
            else:
                stdout = rows
            return {"argv": argv, "returncode": 0, "stdout": stdout, "stderr": ""}
        with patch("omnirsi.execution.repository_identity", return_value={"available": True, "head": "test"}), \
             patch("omnirsi.inspection._command", side_effect=command):
            return inspect_environment(".", "vllm_omni", "cuda", device_ids=ids, device_model=model)

    def test_requested_device_and_model_are_checked(self):
        rows = "0, GPU-A, NVIDIA A100, 999.0\n1, GPU-B, NVIDIA H200, 999.0\n"
        result = self.inspect_cuda(rows, ids=("1",), model="H200")
        self.assertEqual(result["device"]["status"], "available")
        self.assertEqual(result["device"]["devices"][1]["driver_version"], "999.0")
        self.assertEqual(result["compatibility"], "unverified")
        mismatch = self.inspect_cuda(rows, ids=("0",), model="H200")
        self.assertEqual(mismatch["device"]["status"], "device_model_mismatch")

    def test_successful_tool_without_requested_gpu_is_not_available(self):
        empty = self.inspect_cuda("")
        self.assertEqual(empty["device"]["status"], "unavailable")
        missing = self.inspect_cuda("0, GPU-A, NVIDIA H200, 999.0\n", ids=("8",))
        self.assertFalse(missing["device"]["selection_verified"])
        self.assertEqual(missing["device"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
