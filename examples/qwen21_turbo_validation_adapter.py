"""Adapt the historical frozen v4 quality CLI's run-root environment.

The validator itself remains unchanged. v4 interprets OMNIRSI_RESULT_PATH's
parent as the run root, whereas the runner supplies RUN/validation/result.json.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def validation_environment(result_path, environ):
    result = Path(result_path).resolve()
    if result.parent.name != "validation":
        raise ValueError("Expected RUN/validation/result.json")
    root = result.parent.parent
    lock = json.loads((root / "run.lock.json").read_text())
    if lock["repetitions"] != 5:
        raise ValueError("Five complete AB/BA groups required")
    return {**environ, "OMNIRSI_RESULT_PATH": str(root / "v4-root-adapter.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--output", required=True)
    args, remaining = parser.parse_known_args()
    env = validation_environment(os.environ["OMNIRSI_RESULT_PATH"], os.environ)
    command = [sys.executable, args.validator, *remaining, "--output", args.output]
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        return completed.returncode
    result = json.loads(Path(args.output).read_text())
    reports = result.get("reports", [])
    if (
        result.get("verdict") != "PASS"
        or len(reports) != 10
        or any(len(report.get("comparisons", [])) != 4 for report in reports)
    ):
        result.update(
            verdict="FAIL", error="Expected all 40 images across ten complete trials"
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
