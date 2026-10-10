"""Synthetic JSON fixture writer for CLI smoke tests, not performance evidence."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--value", type=float, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
Path(args.output).write_text(json.dumps({"metrics": {"latency_ms": args.value},
                                      "synthetic": True, "note": "No real inference was measured"}), encoding="utf-8")
