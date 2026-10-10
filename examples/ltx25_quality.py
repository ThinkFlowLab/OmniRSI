"""Fail-closed LTX-2.5 artifact guard using stdlib, ffprobe and ffmpeg.

Thresholds are this experiment's declared policy, not an official LTX standard.
The primary mode compares every baseline/candidate trial to one frozen manifest.
"""

import argparse
from array import array
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


class QualityError(ValueError):
    pass


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualityError(f"JSON object required: {path}")
    return value


def load_manifest(path, expected_artifacts=3):
    path = Path(path).resolve()
    manifest = _json(path)
    if manifest.get("status") != "completed" or manifest.get("diagnostic", False):
        raise QualityError(f"completed non-diagnostic trial required: {path}")
    helper_hash = manifest.get("helper_sha256")
    semantic_hashes = manifest.get("semantic_sources_hashes")
    if not isinstance(helper_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", helper_hash):
        raise QualityError(f"frozen helper SHA-256 required: {path}")
    if not isinstance(semantic_hashes, dict) or not semantic_hashes or any(
            not isinstance(name, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in semantic_hashes.items()):
        raise QualityError(f"frozen semantic source hashes required: {path}")
    digest = manifest.get("protocol_sha256")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise QualityError(f"protocol and protocol_sha256 are required: {path}")
    if protocol_digest(protocol) != digest:
        raise QualityError(f"protocol digest mismatch: {path}")
    for field in ("width", "height", "num_frames", "audio_sample_rate", "audio_channels"):
        _number(protocol.get(field), f"protocol {field}", integer=True)
    try:
        fps = Fraction(str(protocol["fps"]))
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise QualityError("protocol fps must be positive") from error
    if fps <= 0 or protocol["audio_sample_rate"] != 48000 or protocol["audio_channels"] != 2:
        raise QualityError("protocol must declare positive fps and 48kHz stereo audio")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len(entries) != expected_artifacts:
        raise QualityError(f"expected exactly {expected_artifacts} artifacts: {path}")
    artifacts, seen_paths = {}, set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise QualityError(f"artifact object required: {path}")
        identifier, filename, recorded_hash = entry.get("id"), entry.get("path"), entry.get("sha256")
        if not isinstance(identifier, str) or not identifier or identifier in artifacts:
            raise QualityError(f"missing or duplicate artifact ID: {path}")
        if not isinstance(filename, str) or not filename:
            raise QualityError(f"artifact path required: {identifier}")
        artifact = Path(filename)
        artifact = (path.parent / artifact).resolve() if not artifact.is_absolute() else artifact.resolve()
        if not artifact.is_file() or artifact.suffix.lower() != ".mp4":
            raise QualityError(f"MP4 artifact missing: {artifact}")
        if artifact in seen_paths:
            raise QualityError(f"duplicate artifact path: {artifact}")
        if not isinstance(recorded_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise QualityError(f"artifact SHA-256 required: {identifier}")
        if sha256_file(artifact) != recorded_hash:
            raise QualityError(f"artifact SHA-256 mismatch: {artifact}")
        artifacts[identifier] = {"path": artifact, "sha256": recorded_hash}
        seen_paths.add(artifact)
    return {"source": str(path), "protocol_sha256": digest, "protocol": protocol, "artifacts": artifacts,
            "helper_sha256": helper_hash, "semantic_sources_hashes": semantic_hashes,
            "dependency_versions": manifest.get("dependency_versions", {})}


def load_directory(directory, expected_artifacts=3):
    """Convenience mode: recursive MP4 discovery rejects duplicate basenames."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise QualityError(f"artifact directory missing: {root}")
    if (root / "result.json").is_file():
        return load_manifest(root / "result.json", expected_artifacts)
    artifacts = {}
    for path in sorted(root.rglob("*.mp4")):
        if path.stem in artifacts:
            raise QualityError(f"duplicate artifact basename under {root}: {path.stem}")
        artifacts[path.stem] = {"path": path.resolve(), "sha256": sha256_file(path)}
    if len(artifacts) != expected_artifacts:
        raise QualityError(f"expected exactly {expected_artifacts} MP4 files under {root}")
    return {"source": str(root), "protocol_sha256": None, "protocol": None, "artifacts": artifacts}


def run_manifests(root):
    """Select exact trial directories; never merge repeated basename maps."""
    root = Path(root).resolve()
    repetitions = _json(root / "run.lock.json").get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise QualityError("run.lock.json must declare positive repetitions")
    expected = {f"trial-{index:03d}" for index in range(1, repetitions + 1)}
    manifests = []
    for side in ("baseline", "candidate"):
        directory = root / side
        found = {path.name for path in directory.glob("trial-*") if path.is_dir()}
        if found != expected:
            raise QualityError(f"{side} trial coverage mismatch: expected {sorted(expected)}, found {sorted(found)}")
        for name in sorted(expected):
            trial = directory / name
            record = _json(trial / "execution.json")
            if record.get("status") != "completed" or record.get("returncode") != 0:
                raise QualityError(f"trial execution did not complete successfully: {trial}")
            result = trial / "result.json"
            if not result.is_file():
                raise QualityError(f"trial result manifest missing: {result}")
            manifests.append((f"{side}/{name}", result))
    return manifests


def _run(argv, evidence, label, timeout):
    (evidence / f"{label}.argv.json").write_text(json.dumps(argv, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = subprocess.run(argv, cwd=evidence, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False, shell=False)
    except subprocess.TimeoutExpired as error:
        (evidence / f"{label}.stderr.log").write_bytes(error.stderr or b"")
        raise QualityError(f"tool timed out after {timeout}s: {argv[0]}") from error
    (evidence / f"{label}.stdout.log").write_bytes(result.stdout)
    (evidence / f"{label}.stderr.log").write_bytes(result.stderr)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise QualityError(f"{argv[0]} exited {result.returncode}: {message}")
    return result


def _number(value, label, integer=False):
    if integer and (isinstance(value, bool) or not isinstance(value, (str, int)) or
                    not re.fullmatch(r"[0-9]+", str(value))):
        raise QualityError(f"{label} must be an integer")
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QualityError(f"invalid {label}: {value}") from error
    if isinstance(value, bool) or not math.isfinite(converted) or converted <= 0:
        raise QualityError(f"{label} must be finite and positive")
    return converted


def parse_probe(value):
    if not isinstance(value, dict):
        raise QualityError("ffprobe JSON object required")
    streams = value.get("streams")
    if not isinstance(streams, list) or any(not isinstance(stream, dict) for stream in streams):
        raise QualityError("ffprobe streams missing")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1 or len(streams) != 2:
        raise QualityError("exactly one video and one audio stream required")
    video, audio = videos[0], audios[0]
    try:
        fps = Fraction(video["avg_frame_rate"])
        nominal_fps = Fraction(video["r_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise QualityError("invalid video frame rate") from error
    if fps <= 0 or nominal_fps <= 0:
        raise QualityError("video frame rate must be positive")
    parsed = {"width": _number(video.get("width"), "width", True),
              "height": _number(video.get("height"), "height", True),
              "frames": _number(video.get("nb_read_frames"), "counted video frames", True),
              "fps": str(fps), "nominal_fps": str(nominal_fps),
              "video_duration": _number(video.get("duration"), "video duration"),
              "audio_duration": _number(audio.get("duration"), "audio duration"),
              "sample_rate": _number(audio.get("sample_rate"), "sample rate", True),
              "channels": _number(audio.get("channels"), "channels", True)}
    for name, stream, field in (("video_codec", video, "codec_name"), ("pixel_format", video, "pix_fmt"),
                                ("audio_codec", audio, "codec_name"), ("channel_layout", audio, "channel_layout")):
        if not isinstance(stream.get(field), str) or not stream[field]:
            raise QualityError(f"ffprobe field missing: {field}")
        parsed[name] = stream[field]
    for name, stream in (("video_start", video), ("audio_start", audio)):
        try:
            parsed[name] = float(stream["start_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise QualityError(f"ffprobe field missing/invalid: {name}") from error
        if not math.isfinite(parsed[name]):
            raise QualityError(f"nonfinite {name}")
    return parsed


def _probe(path, args, evidence, label):
    result = _run([args.ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
                   "-of", "json", str(path)], evidence, label, args.tool_timeout)
    if result.stderr.strip():
        raise QualityError("ffprobe reported decode errors even though its exit status was zero")
    return parse_probe(json.loads(result.stdout))


def compare_metadata(reference, candidate):
    for key in reference:
        if key.endswith("duration") or key.endswith("start"):
            matched = math.isclose(reference[key], candidate[key], rel_tol=0, abs_tol=0.001)
        else:
            matched = reference[key] == candidate[key]
        if not matched:
            raise QualityError(f"media metadata mismatch for {key}: {reference[key]} vs {candidate[key]}")


def check_protocol_metadata(metadata, protocol):
    if protocol is None:
        return
    expected = {"width": int(protocol["width"]), "height": int(protocol["height"]),
                "frames": int(protocol["num_frames"]), "fps": str(Fraction(str(protocol["fps"]))),
                "nominal_fps": str(Fraction(str(protocol["fps"]))),
                "sample_rate": int(protocol["audio_sample_rate"]), "channels": int(protocol["audio_channels"])}
    for key, value in expected.items():
        if metadata[key] != value:
            raise QualityError(f"decoded {key} does not match frozen protocol: {metadata[key]} vs {value}")
    expected_duration = expected["frames"] / float(Fraction(expected["fps"]))
    # Two video frames or two AAC frames cover latent-grid/codec padding only.
    tolerance = max(2 / float(Fraction(expected["fps"])), 2048 / expected["sample_rate"])
    for kind in ("video", "audio"):
        if abs(metadata[kind + "_duration"] - expected_duration) > tolerance:
            raise QualityError(f"{kind} duration does not match frozen protocol: {metadata[kind + '_duration']} vs {expected_duration}")
        if abs(metadata[kind + "_start"]) > tolerance:
            raise QualityError(f"{kind} start offset violates frozen protocol")
    if abs(metadata["video_start"] - metadata["audio_start"]) > tolerance:
        raise QualityError("A/V start synchronization violates frozen protocol")


def check_pcm_duration(samples, protocol):
    if protocol is None:
        return
    fps = float(Fraction(str(protocol["fps"])))
    expected = int(protocol["num_frames"]) / fps
    actual = len(samples) / (int(protocol["audio_channels"]) * int(protocol["audio_sample_rate"]))
    tolerance = max(2 / fps, 2048 / int(protocol["audio_sample_rate"]))
    if abs(actual - expected) > tolerance:
        raise QualityError(f"decoded audio duration does not match frozen protocol: {actual} vs {expected}")


def video_stats(path, key, frames, allow_infinity=False):
    values = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = dict(re.findall(r"([A-Za-z_]+):([^\s]+)", line))
        if fields.get("n") != str(index) or key not in fields:
            raise QualityError(f"missing or unordered per-frame {key} evidence")
        try:
            value = float(fields[key])
        except ValueError as error:
            raise QualityError(f"malformed {key} evidence") from error
        if (not math.isfinite(value) and not (allow_infinity and value == math.inf)) or value < 0:
            raise QualityError(f"invalid per-frame {key}: {value}")
        if key == "All" and value > 1:
            raise QualityError(f"invalid per-frame SSIM: {value}")
        values.append(value)
    if len(values) != frames:
        raise QualityError(f"per-frame {key} count mismatch: expected {frames}, found {len(values)}")
    return values


def audio_nrmse(reference, candidate):
    if not reference or len(reference) != len(candidate) or len(reference) % 2:
        raise QualityError("decoded 48kHz stereo PCM length mismatch or empty output")
    if any(not math.isfinite(sample) for sample in reference) or any(not math.isfinite(sample) for sample in candidate):
        raise QualityError("nonfinite decoded PCM sample")
    channels = []
    for index in range(2):
        before, after = reference[index::2], candidate[index::2]
        reference_energy = math.fsum(sample * sample for sample in before)
        error_energy = math.fsum((right - left) ** 2 for left, right in zip(before, after))
        reference_rms = math.sqrt(reference_energy / len(before))
        silent = reference_rms < 1e-8
        if silent and error_energy != 0:
            raise QualityError(f"silent or near-silent reference channel {index} requires exactly matching PCM")
        nrmse = 0.0 if silent else math.sqrt(error_energy / reference_energy)
        if not math.isfinite(nrmse):
            raise QualityError("nonfinite audio NRMSE")
        channels.append({"channel": index, "nrmse": nrmse, "reference_rms": reference_rms,
                         "silent_reference": silent, "samples": len(before)})
    return {"nrmse": max(channel["nrmse"] for channel in channels),
            "aggregation": "maximum channel NRMSE", "channels": channels, "samples": len(reference)}


def _pcm(path, destination, args, evidence, label):
    _run([args.ffmpeg, "-v", "error", "-nostdin", "-y", "-xerror", "-err_detect", "explode",
          "-i", str(path), "-map", "0:a:0", "-vn", "-ar", "48000", "-ac", "2",
          "-c:a", "pcm_f32le", "-f", "f32le", str(destination)], evidence, label, args.tool_timeout)
    raw = destination.read_bytes()
    if len(raw) % 4:
        raise QualityError("malformed float32 PCM output")
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def compare_pair(reference, candidate, args, evidence, protocol=None):
    evidence.mkdir(parents=True, exist_ok=False)
    ref_path, cand_path = reference["path"], candidate["path"]
    for item in (reference, candidate):
        if sha256_file(item["path"]) != item["sha256"]:
            raise QualityError(f"artifact changed after manifest load: {item['path']}")
    before = _probe(ref_path, args, evidence, "reference-probe")
    after = _probe(cand_path, args, evidence, "candidate-probe")
    check_protocol_metadata(before, protocol)
    check_protocol_metadata(after, protocol)
    compare_metadata(before, after)
    fps = Fraction(before["fps"])
    ordinal_pts = f"N*{fps.denominator}/{fps.numerator}/TB"
    graph = (f"[0:v:0]settb=AVTB,setpts={ordinal_pts},split[r0][r1];"
             f"[1:v:0]settb=AVTB,setpts={ordinal_pts},split[c0][c1];"
             "[r0][c0]ssim=stats_file=ssim.log:shortest=1:repeatlast=0[s];"
             "[r1][c1]psnr=stats_file=psnr.log:shortest=1:repeatlast=0[p]")
    _run([args.ffmpeg, "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode", "-i", str(ref_path),
          "-err_detect", "explode", "-i", str(cand_path), "-filter_complex", graph,
          "-map", "[s]", "-map", "[p]", "-an", "-f", "null", "-"], evidence, "video-compare", args.tool_timeout)
    ssim = video_stats(evidence / "ssim.log", "All", before["frames"])
    psnr = video_stats(evidence / "psnr.log", "psnr_avg", before["frames"], allow_infinity=True)
    pcm_before = _pcm(ref_path, evidence / "reference.f32", args, evidence, "reference-audio")
    pcm_after = _pcm(cand_path, evidence / "candidate.f32", args, evidence, "candidate-audio")
    check_pcm_duration(pcm_before, protocol)
    check_pcm_duration(pcm_after, protocol)
    audio = audio_nrmse(pcm_before, pcm_after)
    minimum_ssim, minimum_psnr = min(ssim), min(psnr)
    failures = []
    if minimum_ssim < args.min_ssim:
        failures.append(f"minimum frame SSIM {minimum_ssim} < {args.min_ssim}")
    if minimum_psnr < args.min_psnr:
        failures.append(f"minimum frame PSNR {minimum_psnr} < {args.min_psnr} dB")
    if audio["nrmse"] > args.max_audio_nrmse:
        failures.append(f"audio NRMSE {audio['nrmse']} > {args.max_audio_nrmse}")
    for item in (reference, candidate):
        if sha256_file(item["path"]) != item["sha256"]:
            failures.append(f"artifact changed during comparison: {item['path']}")
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures,
            "reference_path": str(ref_path), "candidate_path": str(cand_path),
            "reference_sha256": reference["sha256"], "candidate_sha256": candidate["sha256"],
            "metadata": before, "min_frame_ssim": minimum_ssim, "mean_frame_ssim": math.fsum(ssim) / len(ssim),
            "min_frame_psnr_db": "inf" if minimum_psnr == math.inf else minimum_psnr,
            "all_frames_identical_by_psnr": all(value == math.inf for value in psnr),
            "audio": audio, "evidence_dir": str(evidence)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    reference = result.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-manifest", type=Path)
    reference.add_argument("--reference-dir", type=Path)
    candidate = result.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-manifest", type=Path)
    candidate.add_argument("--candidate-dir", type=Path)
    candidate.add_argument("--candidate-root", type=Path)
    candidate.add_argument("--omnirsi-run", action="store_true", help="Derive run root from the validation job result path")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--min-ssim", type=float, default=0.99)
    result.add_argument("--min-psnr", type=float, default=35.0)
    result.add_argument("--max-audio-nrmse", type=float, default=0.01)
    result.add_argument("--expected-artifacts", type=int, default=3)
    result.add_argument("--tool-timeout", type=float, default=300)
    result.add_argument("--ffprobe", default="ffprobe")
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--expected-validator-sha", help="Freeze this guard before Codex changes candidate code")
    result.add_argument("--expected-parallel", help="Frozen TP,Ulysses,VAE tuple for final baseline/candidate trials")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = {"schema_version": 1, "verdict": "FAIL", "errors": [], "comparisons": [],
              "validator_sha256": sha256_file(__file__),
              "thresholds": {"min_frame_ssim": args.min_ssim, "min_frame_psnr_db": args.min_psnr,
                             "max_audio_nrmse": args.max_audio_nrmse,
                             "policy": "Experiment-declared thresholds; not an official LTX-2.5 standard."}}
    exit_code = 1
    output_is_input = False

    def protect_output(paths):
        nonlocal output_is_input
        if args.output.resolve() in {Path(path).resolve() for path in paths}:
            output_is_input = True
            raise QualityError("quality output must not overwrite an input manifest or media artifact")

    try:
        expected_parallel = None
        if args.expected_parallel:
            values = args.expected_parallel.split(",")
            if len(values) != 3 or any(not value.isdigit() for value in values):
                raise QualityError("--expected-parallel requires TP,Ulysses,VAE integers")
            tp, ulysses, vae = map(int, values)
            if tp * ulysses != 4 or vae not in (1, 2, 4):
                raise QualityError("--expected-parallel must describe the frozen four-worker winner")
            expected_parallel = {"tp": tp, "ulysses": ulysses, "vae": vae, "cfg": 1, "ring": 1}
        if args.expected_validator_sha and args.expected_validator_sha != report["validator_sha256"]:
            raise QualityError("validator changed after policy freeze")
        if args.omnirsi_run:
            result_path = Path(os.environ.get("OMNIRSI_RESULT_PATH", "")).resolve()
            if result_path.name != "result.json" or result_path.parent.name != "validation":
                raise QualityError("--omnirsi-run requires the OmniRSI validation job result path")
            args.candidate_root = result_path.parent.parent
        for name in ("min_ssim", "min_psnr", "max_audio_nrmse", "tool_timeout"):
            if not math.isfinite(getattr(args, name)):
                raise QualityError(f"{name} must be finite")
        if not 0 <= args.min_ssim <= 1 or args.min_psnr < 0 or args.max_audio_nrmse < 0:
            raise QualityError("invalid quality threshold range")
        if args.tool_timeout <= 0 or args.expected_artifacts < 1:
            raise QualityError("tool timeout and expected artifact count must be positive")
        reference = (load_manifest(args.reference_manifest, args.expected_artifacts) if args.reference_manifest else
                     load_directory(args.reference_dir, args.expected_artifacts))
        protect_output([reference["source"], *(artifact["path"] for artifact in reference["artifacts"].values())])
        report["reference"] = {"source": reference["source"], "protocol_sha256": reference["protocol_sha256"]}
        if args.candidate_root:
            if reference["protocol_sha256"] is None:
                raise QualityError("candidate-root mode requires a frozen reference manifest with protocol digest")
            candidates = run_manifests(args.candidate_root)
            protect_output([args.candidate_root / "run.lock.json"])
        else:
            candidates = [("direct", args.candidate_manifest)] if args.candidate_manifest else [("directory", None)]
        protect_output([manifest for _, manifest in candidates if manifest is not None])
        evidence_root = args.output.resolve().parent / (args.output.stem + "-evidence-" + uuid.uuid4().hex[:8])
        evidence_root.mkdir(parents=True)
        report["evidence_dir"] = str(evidence_root)
        seen_paths = set()
        for label, manifest in candidates:
            record = {"trial": label, "verdict": "FAIL", "pairs": [], "errors": []}
            report["comparisons"].append(record)
            try:
                candidate = (load_manifest(manifest, args.expected_artifacts) if manifest else
                             load_directory(args.candidate_dir, args.expected_artifacts))
                if expected_parallel:
                    recorded_parallel = _json(manifest).get("parallel", {}) if manifest else {}
                    if any(recorded_parallel.get(key) != value for key, value in expected_parallel.items()):
                        raise QualityError("trial parallel configuration differs from the frozen winner")
                protect_output([candidate["source"], *(artifact["path"] for artifact in candidate["artifacts"].values())])
                record["source"] = candidate["source"]
                if reference["protocol_sha256"] != candidate["protocol_sha256"]:
                    raise QualityError("reference/candidate protocol digest mismatch")
                for field in ("helper_sha256", "semantic_sources_hashes"):
                    if reference.get(field) != candidate.get(field):
                        raise QualityError(f"reference/candidate {field} mismatch")
                def dependencies(value):
                    return {key: version for key, version in value.get("dependency_versions", {}).items()
                            if key != "vllm-omni"}
                if dependencies(reference) != dependencies(candidate):
                    raise QualityError("reference/candidate external dependency versions mismatch")
                if set(reference["artifacts"]) != set(candidate["artifacts"]):
                    raise QualityError("reference/candidate artifact ID set mismatch")
                for identifier, artifact in candidate["artifacts"].items():
                    if args.candidate_root and not artifact["path"].is_relative_to(Path(manifest).resolve().parent):
                        raise QualityError(f"trial artifact escapes its trial directory: {artifact['path']}")
                    if artifact["path"] in seen_paths:
                        raise QualityError(f"artifact path reused across trials: {artifact['path']}")
                    seen_paths.add(artifact["path"])
                    # ID is data: a digest keeps evidence paths independent of arbitrary strings.
                    pair_dir = evidence_root / label / hashlib.sha256(identifier.encode()).hexdigest()[:16]
                    pair = {"id": identifier, "verdict": "FAIL", "evidence_dir": str(pair_dir)}
                    record["pairs"].append(pair)
                    try:
                        pair.update(compare_pair(reference["artifacts"][identifier], artifact, args, pair_dir,
                                                 reference["protocol"]))
                    except (OSError, ValueError, subprocess.SubprocessError, OverflowError) as error:
                        pair["error"] = str(error)
                if record["pairs"] and all(pair["verdict"] == "PASS" for pair in record["pairs"]):
                    record["verdict"] = "PASS"
            except (OSError, ValueError, subprocess.SubprocessError, OverflowError) as error:
                record["errors"].append(str(error))
        if report["comparisons"] and all(record["verdict"] == "PASS" for record in report["comparisons"]):
            report["verdict"], exit_code = "PASS", 0
    except (OSError, ValueError, subprocess.SubprocessError, OverflowError) as error:
        report["errors"].append(str(error))
    except KeyboardInterrupt:
        report["errors"].append("quality validation interrupted")
        exit_code = 130
    # Invalid NaN CLI values are represented as strings so even the error report is valid JSON.
    for key, value in report["thresholds"].items():
        if isinstance(value, float) and not math.isfinite(value):
            report["thresholds"][key] = str(value)
    if output_is_input:
        print(json.dumps({"verdict": "FAIL", "error": "output overlaps an input; inputs were preserved"}))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "report": str(args.output.resolve())}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
