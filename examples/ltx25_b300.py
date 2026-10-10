"""B300 LTX-2.5 campaign helper: fresh servers, full-shape warmup, sync A/V.

No inference library is imported. Run with the prepared Omni interpreter.
All input settings are CLI args; JSON is derived evidence, never a task file.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
from statistics import mean, median
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
MODEL_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
PIPELINE = "LTX2DistilledTwoStagePipeline"
PROMPTS = (
    "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot.",
    "A close shot of a robot playing a piano, its fingers pressing individual keys while the camera remains fixed, with synchronized piano sounds.",
    "A fixed wide shot of ocean waves rolling onto a rocky coast at sunset, with natural surf sounds and steady motion.",
)

# Whole-file hashes freeze the recipe, sigma/guidance and decoder defaults.
SEMANTIC_SOURCE_PATHS = tuple("vllm_omni/diffusion/models/ltx2/" + name for name in (
    "ltx2_recipes.py", "ltx2_guidance.py", "ltx2_request.py", "ltx2_components.py",
    "ltx2_denoise.py", "ltx2_latents.py", "ltx2_diffusion_decoder.py", "ltx2_diffusion_decoder_distributed.py",
    "ltx2_runtime.py", "ltx2_phase_adapter.py", "pipeline_ltx2.py", "pipeline_ltx2_two_stage.py", "ltx2_adapter_parser.py",
)) + ("vllm_omni/model_extras/ltx2.py", "vllm_omni/entrypoints/openai/serving_video.py",
      "vllm_omni/diffusion/utils/media_utils.py")
STAGE_1_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
VIDEO_CODEC_OPTIONS = {"preset": "ultrafast", "threads": "0", "crf": "18"}


def protocol(args=None):
    width, height = getattr(args, "width", 1280), getattr(args, "height", 704)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 64
           for value in (width, height)):
        raise ValueError("Two-stage width and height must be positive multiples of 64")
    stage_1, stage_2 = STAGE_1_SIGMAS, STAGE_2_SIGMAS
    if getattr(args, "diagnostic", False):
        schedules = []
        for sigmas, steps in ((stage_1, args.profile_stage1_steps), (stage_2, args.profile_stage2_steps)):
            if not 1 <= steps < len(sigmas):
                raise ValueError("Diagnostic steps must be within the fixed stage schedule")
            schedules.append(tuple(sigmas[i * (len(sigmas) - 1) // steps] for i in range(steps + 1)))
        stage_1, stage_2 = schedules
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "pipeline": PIPELINE,
            "task": "t2v", "width": width, "height": height, "num_frames": 121, "fps": 24,
            "num_inference_steps": len(stage_1) - 1, "refine_steps": len(stage_2) - 1, "dtype": "bfloat16",
            "decoder": "native_diffvae", "audio_sample_rate": 48000, "audio_channels": 2,
            "stage_1_sigmas": list(stage_1), "stage_2_sigmas": list(stage_2),
            "guidance": "positive_only", "ltx2_use_conv_vae": False,
            "video_codec_options": dict(VIDEO_CODEC_OPTIONS),
            "prompts": [{"id": f"prompt-{i:03d}-seed-{42+i}", "prompt": prompt, "seed": 42+i}
                        for i, prompt in enumerate(PROMPTS)], "warmup_rounds": 2,
            "warmup_seeds": [[42+i+10000+warmup*1000 for i in range(len(PROMPTS))] for warmup in range(2)],
            "request_scope": "New synchronous requests; distinct warmup seeds; static conditioning reuse is allowed."}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_parallel(tp, ulysses, vae, single_reference=False):
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (tp, ulysses, vae)):
        raise ValueError("Parallel degrees must be positive integers")
    if (tp, ulysses, vae) != (1, 1, 1) and single_reference:
        raise ValueError("The single-worker reference requires TP1 / Ulysses1 / VAE1")
    if not single_reference and ((tp, ulysses) not in ((1, 4), (2, 2), (4, 1)) or vae not in (1, 2, 4)):
        raise ValueError("Four-worker trials permit TP1/U4, TP2/U2 or TP4/U1 with VAE1,2,4; CFG/Ring stay one")
    world = tp * ulysses
    if world != (1 if single_reference else 4):
        raise ValueError("TP * Ulysses must equal four; only the explicit reference lane uses one")
    if vae > world or world % vae:
        raise ValueError("VAE degree must divide and reuse the existing worker WORLD")
    return world


def storage_locations(args, output_root, require_source=True):
    source = Path(args.source_repo).resolve()
    cache = Path(args.cache_root).resolve()
    output_root = Path(output_root).resolve()
    if require_source and not source.is_dir():
        raise ValueError("source-repo must be an existing checkout")
    if cache.is_relative_to(source) or output_root.is_relative_to(source):
        raise ValueError("Cache and campaign output must be outside the source checkout")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(key) and Path(os.environ[key]).resolve().is_relative_to(source):
            raise ValueError(f"{key} must be outside the source checkout")
    return source, cache, output_root


def source_snapshot(source):
    from omnirsi.execution import repository_identity
    identity = repository_identity(source)
    if not identity.get("available"):
        raise ValueError(f"Source identity unavailable: {identity.get('error')}")
    return identity


def semantic_sources(source):
    source = Path(source).resolve()
    if not SEMANTIC_SOURCE_PATHS:
        raise ValueError("No reviewed semantic source paths were declared; campaign is unqualified")
    hashes = {}
    for relative in SEMANTIC_SOURCE_PATHS:
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Required semantic source missing or outside checkout: {relative}")
        hashes[relative] = sha256(path)
    return hashes


def dependency_versions():
    packages = ("torch", "vllm", "vllm-omni", "diffusers", "transformers", "triton",
                "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13", "nvidia-cudnn-cu12",
                "nvidia-cudnn-cu13", "nvidia-nccl-cu12", "nvidia-nccl-cu13", "flashinfer-python")
    code = ("import importlib.metadata as metadata,json,sys\n"
            "versions={}\n"
            "for name in sys.argv[1:]:\n"
            " try: versions[name]=metadata.version(name)\n"
            " except metadata.PackageNotFoundError: versions[name]='not_installed'\n"
            "print(json.dumps(versions))\n")
    result = subprocess.run([sys.executable, "-c", code, *packages], capture_output=True, text=True,
                            timeout=30, check=True)
    return json.loads(result.stdout)


def semantic_dependency_versions(versions):
    # Source revisions cover vLLM-Omni code; editable package version text can differ.
    return {name: version for name, version in versions.items() if name != "vllm-omni"}


def server_argv(args):
    world = validate_parallel(args.tp, args.ulysses, args.vae, args.single_reference)
    executable = str(Path(sys.executable).parent / "vllm")
    argv = [executable, "serve", args.model, "--revision", MODEL_REVISION, "--omni",
            "--model-class-name", PIPELINE, "--dtype", "bfloat16", "--num-gpus", str(world),
            "--tensor-parallel-size", str(args.tp), "--usp", str(args.ulysses), "--ring", "1",
            "--cfg-parallel-size", "1", "--vae-patch-parallel-size", str(args.vae),
            "--vae-parallel-mode", "tile", "--vae-use-tiling",
            "--diffusion-attention-backend", "CUDNN_ATTN", "--max-num-seqs", "1",
            "--stage-overrides", json.dumps({"0": {"extras": {"ltx2_use_conv_vae": False}}}),
            "--host", "127.0.0.1", "--port", str(args.port),
            "--stage-init-timeout", str(args.startup_timeout)]
    if args.eager:
        argv.append("--enforce-eager")
    if args.diagnostic:
        directory = str(Path(args.output).resolve().parent / "torch-profile")
        argv.extend(["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": directory})])
    return argv


def preflight():
    ids = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly four caller-reserved B300 IDs/UUIDs")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=30)
    devices = [dict(zip(("index", "uuid", "name", "driver"), [v.strip() for v in row]))
               for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 4]
    selected = [gpu for gpu in devices if gpu["index"] in ids or gpu["uuid"] in ids]
    if len(selected) != 4 or any(not re.search(r"\bB300\b", gpu["name"]) for gpu in selected):
        raise ValueError("The four requested physical devices must all be B300; inspect nvidia-smi mapping")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=30)
    selected_uuids = {gpu["uuid"] for gpu in selected}
    if any(row and row[0].strip() in selected_uuids for row in csv.reader(io.StringIO(active.stdout))):
        raise ValueError("An allocated GPU already has a compute process; do not overlap campaign trials")
    return selected


def multipart(fields):
    boundary = "omnirsi-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), "multipart/form-data; boundary=" + boundary


def request_video(args, item, destination):
    settings = protocol(args)
    fields = {"model": args.model, "prompt": item["prompt"], "seed": item["seed"],
              "width": settings["width"], "height": settings["height"], "num_frames": 121, "fps": 24,
              "num_inference_steps": settings["num_inference_steps"],
              "extra_params": json.dumps({"stage_1_sigmas": settings["stage_1_sigmas"],
                                           "stage_2_sigmas": settings["stage_2_sigmas"],
                                           "video_codec_options": VIDEO_CODEC_OPTIONS})}
    body, content_type = multipart(fields)
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/videos/sync", data=body,
                                     headers={"Content-Type": content_type, "Accept": "video/mp4"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        if response.status != 200 or "video" not in response.headers.get("Content-Type", ""):
            raise ValueError("Synchronous endpoint did not return a successful video artifact")
        count = 0
        with Path(destination).open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                count += len(block)
    elapsed = (time.perf_counter() - start) * 1000
    if not count or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Incomplete or invalid synchronous request")
    return elapsed


def launch_server(argv, source, environment, log):
    """Keep supervised workers inside the runner's group, including hard timeout."""
    helper_pid, helper_group = os.getpid(), os.getpgrp()
    marker = bool(os.environ.get("OMNIRSI_RESULT_PATH"))
    if marker and helper_group != helper_pid:
        raise ValueError("OmniRSI helper must be its trial group leader; invoke the prepared Python directly")
    supervised = marker and helper_group == helper_pid
    process = subprocess.Popen(argv, cwd=source, env=environment, stdout=log, stderr=log,
                               start_new_session=not supervised, shell=False)
    ownership = {"mode": "omnirsi_trial_group" if supervised else "server_session",
                 "server_pid": process.pid, "helper_pid": helper_pid,
                 "process_group_id": helper_group if supervised else process.pid,
                 "helper_process_group_id": helper_group,
                 "server_start_new_session": not supervised}
    return process, ownership


def _live_group_members(group, exclude=()):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            if int(fields[2]) == group and fields[0] != "Z" and pid not in exclude:
                members.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_server(process, ownership, grace_seconds=20):
    """Clean the owned group even when its original server leader already exited."""
    group = ownership["process_group_id"]
    inherited = ownership["mode"] == "omnirsi_trial_group"
    if inherited:
        if group != os.getpid() or group != os.getpgrp() or ownership["helper_pid"] != os.getpid():
            raise ValueError("Refusing cleanup of a group not owned by this supervised helper")
        excluded = (os.getpid(),)
    else:
        if group != process.pid or group == os.getpgrp() or ownership["server_start_new_session"] is not True:
            raise ValueError("Refusing cleanup of an unknown server group")
        excluded = ()

    def signal_members(number):
        if inherited:
            for pid in _live_group_members(group, excluded):
                try:
                    if os.getpgid(pid) == group:
                        os.kill(pid, number)
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(group, number)
            except ProcessLookupError:
                pass

    signal_members(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _live_group_members(group, excluded) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    remaining = _live_group_members(group, excluded)
    killed = bool(remaining)
    if remaining:
        signal_members(signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _live_group_members(group, excluded) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.05)
    if process.poll() is None:
        process.wait(timeout=5)
    remaining = _live_group_members(group, excluded)
    return {"status": "failed" if remaining else "completed", "mode": ownership["mode"], "process_group_id": group, "term_sent": True,
            "kill_sent": killed, "remaining_pids": remaining,
            "server_returncode": process.returncode}


def run_trial(args):
    if sys.platform != "linux":
        raise ValueError("GPU campaign execution requires the prepared remote Linux environment")
    output = Path(args.output).resolve()
    validate_parallel(args.tp, args.ulysses, args.vae, args.single_reference)
    source, cache, _ = storage_locations(args, output.parent)
    before = source_snapshot(source)
    semantics = semantic_sources(source)
    if output.exists():
        raise ValueError("Result path already exists; keep trials immutable and use a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    media = output.parent / "media"
    media.mkdir(exist_ok=False)
    base_protocol = protocol(args)
    record = {"status": "running", "protocol": base_protocol, "protocol_sha256": digest(base_protocol),
              "parallel": {"tp": args.tp, "ulysses": args.ulysses, "cfg": 1, "ring": 1, "vae": args.vae},
              "compile_policy": "eager" if args.eager else "default", "diagnostic": args.diagnostic,
              "source_repo": str(source), "source_before": before, "source_commit": before["head"],
              "semantic_sources_hashes": semantics, "helper_sha256": sha256(__file__),
              "dependency_version_scope": "Installed distribution metadata; loaded CUDA/cuDNN/NCCL runtime versions are not probed.",
              "tp_qualification": "Requires at least three quality-passing repeats before eligibility" if args.tp != 1 else "TP1",
              "server_argv": server_argv(args), "artifacts": [], "samples_ms": []}
    process, ownership = None, None
    try:
        record["dependency_versions"] = dependency_versions()
        record["devices"] = preflight()
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise ValueError("Campaign port is already in use; do not attach to an unknown server")
        if not Path(record["server_argv"][0]).is_file():
            raise ValueError("No vllm entrypoint beside this interpreter; use the prepared Omni Python")
        cache.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(Path(args.source_repo).resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
        environment.setdefault("HF_HOME", str(cache / "huggingface"))
        for key, directory in {"TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
                               "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "cuda"}.items():
            environment[key] = str(cache / directory)
        with (output.parent / "server.log").open("wb") as log:
            process, ownership = launch_server(record["server_argv"], source, environment, log)
            record["server_process"] = ownership
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise ValueError("Server exited before ready; inspect server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness deadline exceeded")
                time.sleep(2)
            for warmup in range(base_protocol["warmup_rounds"]):
                for index, measured_item in enumerate(base_protocol["prompts"]):
                    item = {**measured_item, "seed": base_protocol["warmup_seeds"][warmup][index]}
                    temporary = media / f"warmup-{warmup}-{item['id']}.mp4"
                    request_video(args, item, temporary)
                    temporary.unlink()
            if args.diagnostic:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{args.port}/start_profile", data=b""), timeout=60).close()
            for item in base_protocol["prompts"]:
                artifact = media / (item["id"] + ".mp4")
                elapsed = request_video(args, item, artifact)
                record["samples_ms"].append({"id": item["id"], "latency_ms": elapsed})
                record["artifacts"].append({"id": item["id"], "path": str(artifact), "sha256": sha256(artifact)})
            if args.diagnostic:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{args.port}/stop_profile", data=b""), timeout=60).close()
            else:
                record["metrics"] = {"latency_ms": mean(item["latency_ms"] for item in record["samples_ms"])}
            record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="cancelled", error="interrupted by user")
    except (OSError, ValueError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
        record.update(status="failed", error=str(exc))
    finally:
        if process is not None:
            try:
                record["server_cleanup"] = stop_server(process, ownership, args.stop_grace_seconds)
                if record["server_cleanup"]["remaining_pids"]:
                    record.update(status="failed", error="Owned server processes remained after cleanup")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                record.update(status="failed", error=f"Server cleanup failed: {error}",
                              server_cleanup={"status": "failed", "remaining_pids": "unknown", "error": str(error)})
        try:
            after = source_snapshot(source)
            record["source_after"] = after
            record["source_integrity"] = {"verified": before["snapshot_hash"] == after["snapshot_hash"],
                                          "dirty_at_start": before["dirty"]}
            if not record["source_integrity"]["verified"] or semantic_sources(source) != semantics:
                record.update(status="failed", error="Source changed during trial")
            if sha256(__file__) != record["helper_sha256"]:
                record.update(status="failed", error="Campaign helper changed during trial")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record.update(status="failed", error=f"Source integrity check failed: {error}",
                          source_integrity={"verified": False, "dirty_at_start": before["dirty"]})
        write_json(output, record)
    return record


class CampaignDrift(ValueError):
    """A frozen campaign input changed: abort, rather than rank mixed evidence."""


def freeze_campaign(args):
    source, _, _ = storage_locations(args, args.output_dir)
    snapshot = source_snapshot(source)
    if snapshot["dirty"]:
        raise ValueError("Configuration search requires a clean pinned source checkout")
    reference_path = Path(args.reference_manifest).resolve()
    quality_path = Path(args.quality_script).resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or (reference.get("status") != "completed" or reference.get("diagnostic", False) or
            reference.get("protocol") != protocol(args) or reference.get("protocol_sha256") != digest(protocol(args))):
        raise ValueError("Reference must be a completed manifest for this exact frozen protocol")
    semantics = semantic_sources(source)
    if reference.get("semantic_sources_hashes") != semantics or reference.get("helper_sha256") != sha256(__file__):
        raise ValueError("Reference semantic sources/helper do not match this campaign")
    if reference.get("source_commit") != snapshot["head"]:
        raise ValueError("Reference and configuration search must use the same pinned source commit")
    artifacts = reference.get("artifacts", [])
    if (not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts) or
            {item.get("id") for item in artifacts} != {item["id"] for item in protocol(args)["prompts"]} or len(artifacts) != 3):
        raise ValueError("Reference must contain exactly the three fixed prompt/seed artifacts")
    recorded_artifacts, paths = {}, set()
    for item in artifacts:
        path = Path(item["path"])
        path = (reference_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if path in paths or not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError("Reference artifact missing, duplicated or changed")
        recorded_artifacts[item["id"]] = {"path": str(path), "sha256": item["sha256"]}
        paths.add(path)
    versions = dependency_versions()
    if semantic_dependency_versions(reference.get("dependency_versions", {})) != semantic_dependency_versions(versions):
        raise ValueError("Reference and configuration search dependency versions differ")
    return {"source": snapshot, "semantic_sources_hashes": semantics, "dependency_versions": versions,
            "helper_sha256": sha256(__file__), "quality_script": {"path": str(quality_path), "sha256": sha256(quality_path)},
            "reference_manifest": {"path": str(reference_path), "sha256": sha256(reference_path), "artifacts": recorded_artifacts}}


def assert_campaign_frozen(args, frozen):
    try:
        current = source_snapshot(Path(args.source_repo).resolve())
        if current["snapshot_hash"] != frozen["source"]["snapshot_hash"] or current["dirty"]:
            raise CampaignDrift("Source changed during configuration search")
        if semantic_sources(Path(args.source_repo).resolve()) != frozen["semantic_sources_hashes"]:
            raise CampaignDrift("Semantic source files changed during configuration search")
        if semantic_dependency_versions(dependency_versions()) != semantic_dependency_versions(frozen["dependency_versions"]):
            raise CampaignDrift("Installed dependency versions changed during configuration search")
        if sha256(__file__) != frozen["helper_sha256"]:
            raise CampaignDrift("Campaign helper changed during configuration search")
        if sha256(frozen["quality_script"]["path"]) != frozen["quality_script"]["sha256"]:
            raise CampaignDrift("Quality guard changed during configuration search")
        reference = frozen["reference_manifest"]
        if sha256(reference["path"]) != reference["sha256"]:
            raise CampaignDrift("Reference manifest changed during configuration search")
        if any(sha256(item["path"]) != item["sha256"] for item in reference["artifacts"].values()):
            raise CampaignDrift("Reference artifact changed during configuration search")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if isinstance(error, CampaignDrift):
            raise
        raise CampaignDrift(f"Frozen campaign evidence became unavailable: {error}") from error


def search(args):
    if args.repetitions < 1 or args.include_tp_candidates and args.repetitions < 3:
        raise ValueError("Search requires positive repetitions; TP candidates require at least three")
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise ValueError("Search output already exists; start a new campaign directory")
    frozen = freeze_campaign(args)
    root.mkdir(parents=True)
    campaign = {"status": "running", "protocol": protocol(args), "protocol_sha256": digest(protocol(args)),
                "frozen_evidence": frozen, "cells": []}
    try:
        layouts = [(1, 4)] + ([(2, 2), (4, 1)] if args.include_tp_candidates else [])
        for tp, ulysses, vae in ((tp, u, v) for tp, u in layouts for v in (1, 2, 4)):
            cell = {"id": f"tp{tp}-u{ulysses}-v{vae}-default", "tp": tp, "ulysses": ulysses, "vae": vae, "cfg": 1,
                    "ring": 1, "eager": False, "status": "running", "values_ms": [], "trials": []}
            campaign["cells"].append(cell)
            write_json(root / "search.json", campaign)
            try:
                for repetition in range(args.repetitions):
                    assert_campaign_frozen(args, frozen)
                    output = root / cell["id"] / f"trial-{repetition+1:03d}" / "result.json"
                    trial_args = {**vars(args), "tp": tp, "ulysses": ulysses, "vae": vae, "eager": False,
                                  "single_reference": False, "diagnostic": False, "output": str(output)}
                    result = run_trial(argparse.Namespace(**trial_args))
                    assert_campaign_frozen(args, frozen)
                    if result["status"] == "cancelled":
                        raise KeyboardInterrupt
                    if result.get("source_integrity", {}).get("verified") is not True:
                        raise CampaignDrift("A trial source snapshot changed; search is invalid")
                    if result.get("server_cleanup", {}).get("remaining_pids") or result.get("server_cleanup", {}).get("status") == "failed":
                        raise CampaignDrift("Server workers remained; search cannot continue")
                    if result["status"] != "completed":
                        raise ValueError(result.get("error", "trial failed"))
                    quality = output.parent / "quality.json"
                    validation = subprocess.run([sys.executable, frozen["quality_script"]["path"],
                        "--reference-manifest", frozen["reference_manifest"]["path"], "--candidate-manifest", str(output),
                        "--output", str(quality), "--min-ssim", "0.99", "--min-psnr", "35",
                        "--max-audio-nrmse", "0.01", "--expected-validator-sha", frozen["quality_script"]["sha256"]], check=False)
                    assert_campaign_frozen(args, frozen)
                    if validation.returncode != 0 or json.loads(quality.read_text(encoding="utf-8")).get("verdict") != "PASS":
                        raise ValueError("A/V quality gate failed; cell is ineligible")
                    value = result.get("metrics", {}).get("latency_ms")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError("Trial latency must be finite and positive")
                    cell["values_ms"].append(value)
                    cell["trials"].append({"manifest": str(output), "quality_report": str(quality),
                                           "quality_report_sha256": sha256(quality), "latency_ms": value})
                    write_json(root / "search.json", campaign)
                cell.update(status="eligible", median_ms=median(cell["values_ms"]),
                            tp_qualification="three_or_more_quality_passing_repeats" if tp != 1 else "TP1")
            except CampaignDrift:
                raise
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                cell.update(status="ineligible", error=str(error))
            write_json(root / "search.json", campaign)
        assert_campaign_frozen(args, frozen)
        eligible = [cell for cell in campaign["cells"] if cell["status"] == "eligible"]
        if not eligible:
            raise ValueError("No quality-passing configuration; no best baseline can be claimed")
        best = min(eligible, key=lambda cell: cell["median_ms"])
        write_json(root / "best.json", {"scope": "Lowest recorded median among tested quality-passing four-worker cells; not a global or statistically proven optimum.",
                                        "best": best, "frozen_evidence": frozen, "protocol_sha256": digest(protocol())})
        campaign["status"] = "completed"
        write_json(root / "search.json", campaign)
        print(json.dumps(best, indent=2))
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        campaign.update(status="aborted", error="interrupted by user" if isinstance(error, KeyboardInterrupt) else str(error))
        write_json(root / "search.json", campaign)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("trial", "search"):
        child = commands.add_parser(name)
        child.add_argument("--source-repo", required=True)
        child.add_argument("--model", default=MODEL_ID, choices=(MODEL_ID,))
        child.add_argument("--cache-root", required=True)
        child.add_argument("--width", type=int, default=1280)
        child.add_argument("--height", type=int, default=704,
                           help="720p-class default; two-stage dimensions must be multiples of 64")
        child.add_argument("--port", type=int, default=8098)
        child.add_argument("--startup-timeout", type=int, default=1800)
        child.add_argument("--request-timeout", type=int, default=900)
        child.add_argument("--stop-grace-seconds", type=float, default=20)
        child.add_argument("--dry-run", action="store_true")
        if name == "trial":
            child.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
            child.add_argument("--ulysses", type=int, choices=(1, 2, 4), default=4)
            child.add_argument("--vae", type=int, choices=(1, 2, 4), default=4)
            child.add_argument("--single-reference", action="store_true")
            child.add_argument("--eager", action="store_true")
            child.add_argument("--diagnostic", action="store_true")
            child.add_argument("--profile-stage1-steps", type=int, choices=range(1, 9), default=2,
                               help="Diagnostic only: subsample the fixed stage-1 sigma schedule")
            child.add_argument("--profile-stage2-steps", type=int, choices=range(1, 4), default=1,
                               help="Diagnostic only: subsample the fixed stage-2 sigma schedule")
            child.add_argument("--output", required=True)
        else:
            child.add_argument("--reference-manifest", required=True)
            child.add_argument("--quality-script", required=True)
            child.add_argument("--repetitions", type=int, default=3)
            child.add_argument("--include-tp-candidates", action="store_true",
                               help="Test TP2/U2 and TP4/U1; require >=3 repeats and full A/V quality for eligibility")
            child.add_argument("--output-dir", required=True)
    return result


def _termination_handler(_signal, _frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parser().parse_args(argv)
    previous_sigterm = None
    if sys.platform == "linux" and not args.dry_run:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _termination_handler)
    try:
        settings = protocol(args)
        if args.command == "trial" and not args.diagnostic and (args.profile_stage1_steps, args.profile_stage2_steps) != (2, 1):
            raise ValueError("Profile step overrides require --diagnostic; benchmark steps stay 8+3")
        if args.startup_timeout <= 0 or args.request_timeout <= 0 or not 1 <= args.port <= 65535 or not math.isfinite(args.stop_grace_seconds) or args.stop_grace_seconds < 0:
            raise ValueError("Timeouts and port must be valid positive values")
        if args.command == "search" and args.repetitions < 1:
            raise ValueError("Search repetitions must be positive")
        if args.command == "search" and args.include_tp_candidates and args.repetitions < 3:
            raise ValueError("TP candidates require at least three independent quality-passing repeats")
        if args.dry_run:
            storage_locations(args, Path(args.output).resolve().parent if args.command == "trial" else args.output_dir,
                              require_source=False)
            preview = {"protocol": settings, "protocol_sha256": digest(settings), "gpu_execution": "not started",
                       "diagnostic": getattr(args, "diagnostic", False)}
            if args.command == "trial":
                preview["server_argv"] = server_argv(args)
            else:
                layouts = [(1, 4)] + ([(2, 2), (4, 1)] if args.include_tp_candidates else [])
                preview["grid"] = [{"tp": tp, "ulysses": u, "vae": vae, "cfg": 1, "ring": 1}
                                   for tp, u in layouts for vae in (1, 2, 4)]
            preview["semantic_source_paths"] = SEMANTIC_SOURCE_PATHS
            print(json.dumps(preview, indent=2))
            return 0
        if args.command == "search":
            search(args)
            return 0
        record = run_trial(args)
        print(json.dumps({"output": args.output, "status": record["status"]}))
        return 0 if record["status"] == "completed" else 130 if record["status"] == "cancelled" else 1
    except KeyboardInterrupt:
        print("Campaign interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
