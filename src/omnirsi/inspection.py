"""Inspect a prepared local target; never install or import its GPU libraries."""

from __future__ import annotations

import json
import csv
import io
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .catalog import BACKENDS, repository_pack


def inspect_environment(repo: str, repo_type: str, backend: str, *,
                        device_ids: tuple[str, ...] = (), device_model: str | None = None,
                        python: str | None = None) -> dict:
    from .execution import repository_identity

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    repository = repository_identity(Path(repo).resolve())
    python = python or sys.executable
    package_code = (
        "import json,sys,importlib.metadata as m; "
        "names=['torch','vllm','vllm-omni','vllm-afd-plugin','vllm-rlt','torch-npu','nvidia-nccl-cu12','nvidia-nccl-cu13','nvidia-cuda-runtime-cu12','nvidia-cuda-runtime-cu13']; "
        "installed={d.metadata['Name'].lower():d.version for d in m.distributions() if d.metadata['Name']}; "
        "print(json.dumps({'python':sys.version,'packages':{n:installed.get(n) for n in names}}))"
    )
    python_info = _command([python, "-c", package_code])
    if python_info["returncode"] == 0:
        try:
            python_info["environment"] = json.loads(python_info["stdout"])
        except json.JSONDecodeError:
            python_info["parse_error"] = "Target Python did not emit a JSON environment"

    commands = {
        "cuda": ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        "rocm": ["rocminfo"],
        "ascend": ["npu-smi", "info"],
    }
    if backend == "cpu":
        device = {"status": "available", "tool": "platform", "details": platform.processor(),
                  "note": "CPU contract/smoke testing only; no accelerator validation"}
    else:
        probe = _command(commands[backend])
        device = {"status": "tool_available" if probe["returncode"] == 0 else "unavailable",
                  "probe": probe, "selection_verified": False,
                  "note": "Raw tool output is not a model/backend compatibility certification"}
        if backend == "cuda" and probe["returncode"] == 0:
            rows = []
            for row in csv.reader(io.StringIO(probe["stdout"])):
                fields = [value.strip() for value in row]
                if len(fields) == 4 and fields[0].isdigit():
                    rows.append(dict(zip(("index", "uuid", "name", "driver_version"), fields)))
            device["topology"] = _command(["nvidia-smi", "topo", "-m"])
            device["devices"] = rows
            selected = [row for row in rows if not device_ids or row["index"] in device_ids or row["uuid"] in device_ids]
            device["selection_verified"] = bool(selected) and len(selected) == (len(device_ids) if device_ids else len(rows))
            if not device["selection_verified"]:
                device["status"] = "unavailable"
            elif device_model and any(device_model.casefold() not in row["name"].casefold() for row in selected):
                device["status"] = "device_model_mismatch"
            else:
                device["status"] = "available"

    return {
        "host": {"platform": platform.platform(), "hostname": platform.node()},
        "repository": repository,
        "repository_pack": repository_pack(repo_type),
        "backend": backend,
        "requested_device_ids": list(device_ids),
        "requested_device_model": device_model,
        "visible_devices": {key: os.environ.get(key) for key in
                            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")},
        "target_python": python_info,
        "device": device,
        "compatibility": "unverified",
    }


def _command(argv: list[str]) -> dict:
    if not Path(argv[0]).is_file() and shutil.which(argv[0]) is None:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": "Executable not found"}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}
