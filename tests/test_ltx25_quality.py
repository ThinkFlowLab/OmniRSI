"""CPU-only quality policy contracts; no model or accelerator invocation."""

from array import array
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ltx25_quality.py"
SPEC = importlib.util.spec_from_file_location("ltx25_quality", SCRIPT)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def media_metadata():
    return {"streams": [
        {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1920,
         "height": 1088, "nb_read_frames": "3", "avg_frame_rate": "24/1", "r_frame_rate": "24/1",
         "duration": "0.125000", "start_time": "0.000000"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2,
         "channel_layout": "stereo", "duration": "0.125000", "start_time": "0.000000"}]}


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.make_manifest(self.root / "reference")
        self.candidate = self.make_manifest(self.root / "candidate")
        self.output = self.root / "quality.json"
        self.video_calls = []

    def make_manifest(self, directory, protocol=None):
        directory.mkdir(parents=True)
        artifacts = []
        for index in range(3):
            identifier = f"prompt-{index:03d}-seed-42"
            path = directory / f"{identifier}.mp4"
            path.write_bytes(f"synthetic artifact {index}".encode())
            artifacts.append({"id": identifier, "path": str(path), "sha256": quality.sha256_file(path)})
        protocol = {"model": "LTX-2.5", "width": 1920, "height": 1088, "num_frames": 3, "fps": 24,
                    "audio_sample_rate": 48000, "audio_channels": 2, "dtype": "bfloat16", **(protocol or {})}
        result = directory / "result.json"
        result.write_text(json.dumps({"protocol": protocol, "protocol_sha256": quality.protocol_digest(protocol),
                                      "status": "completed", "diagnostic": False,
                                      "helper_sha256": "a" * 64,
                                      "semantic_sources_hashes": {"ltx2_recipes.py": "b" * 64},
                                      "parallel": {"tp": 1, "ulysses": 4, "vae": 4, "cfg": 1, "ring": 1},
                                      "artifacts": artifacts, "metrics": {"latency_ms": 1000}}), encoding="utf-8")
        return result

    def fake_tool(self, argv, evidence, label, timeout):
        if label.endswith("probe"):
            return subprocess.CompletedProcess(argv, 0, json.dumps(media_metadata()).encode(), b"")
        if label == "video-compare":
            self.video_calls.append(argv)
            (evidence / "ssim.log").write_text("".join(f"n:{index} All:1.000000 (inf)\n" for index in range(1, 4)))
            (evidence / "psnr.log").write_text("".join(f"n:{index} psnr_avg:inf\n" for index in range(1, 4)))
        elif label.endswith("audio"):
            self.assertIn("48000", argv)
            self.assertEqual(argv[argv.index("-ac") + 1], "2", "audio must preserve both stereo channels")
            values = array("f", [0.1, 0.2] * 6000)
            if sys.byteorder != "little":
                values.byteswap()
            Path(argv[-1]).write_bytes(values.tobytes())
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def invoke(self, extra=None, fake=None):
        arguments = ["--reference-manifest", str(self.reference), "--candidate-manifest", str(self.candidate),
                     "--output", str(self.output)]
        if extra:
            arguments += extra
        with patch.object(quality, "_run", side_effect=fake or self.fake_tool), redirect_stdout(io.StringIO()):
            code = quality.main(arguments)
        return code, json.loads(self.output.read_text(encoding="utf-8"))

    def test_matching_artifacts_pass_with_positive_infinite_psnr_as_json_string(self):
        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(report["comparisons"][0]["pairs"]), 3)
        pair = report["comparisons"][0]["pairs"][0]
        self.assertEqual(pair["min_frame_psnr_db"], "inf")
        self.assertEqual(pair["audio"]["nrmse"], 0)
        self.assertTrue(pair["all_frames_identical_by_psnr"])
        self.assertNotIn("NaN", self.output.read_text(encoding="utf-8"))

    def test_one_bad_frame_fails_even_when_other_frames_are_perfect(self):
        def tool(argv, evidence, label, timeout):
            result = self.fake_tool(argv, evidence, label, timeout)
            if label == "video-compare":
                (evidence / "ssim.log").write_text("n:1 All:1\nn:2 All:0.98\nn:3 All:1\n")
            return result
        code, report = self.invoke(fake=tool)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertIn("minimum frame SSIM", report["comparisons"][0]["pairs"][0]["failures"][0])

    def test_one_low_psnr_frame_fails(self):
        def tool(argv, evidence, label, timeout):
            result = self.fake_tool(argv, evidence, label, timeout)
            if label == "video-compare":
                (evidence / "psnr.log").write_text("n:1 psnr_avg:inf\nn:2 psnr_avg:34.9\nn:3 psnr_avg:inf\n")
            return result
        code, report = self.invoke(fake=tool)
        self.assertNotEqual(code, 0)
        self.assertIn("minimum frame PSNR", report["comparisons"][0]["pairs"][0]["failures"][0])

    def test_right_audio_channel_regression_cannot_be_hidden_by_mono_conversion(self):
        def tool(argv, evidence, label, timeout):
            result = self.fake_tool(argv, evidence, label, timeout)
            if label == "candidate-audio":
                values = array("f", [0.1, 0.3] * 6000)
                if sys.byteorder != "little":
                    values.byteswap()
                Path(argv[-1]).write_bytes(values.tobytes())
            return result
        code, report = self.invoke(fake=tool)
        self.assertNotEqual(code, 0)
        self.assertGreater(report["comparisons"][0]["pairs"][0]["audio"]["nrmse"], 0.01)

    def test_manifest_hash_protocol_and_duplicates_are_rejected(self):
        original = json.loads(self.candidate.read_text(encoding="utf-8"))
        variants = []
        tampered = copy.deepcopy(original)
        tampered["artifacts"][0]["sha256"] = "0" * 64
        variants.append(tampered)
        altered_protocol = copy.deepcopy(original)
        altered_protocol["protocol"]["frames"] = 99
        variants.append(altered_protocol)
        duplicate_id = copy.deepcopy(original)
        duplicate_id["artifacts"][1]["id"] = duplicate_id["artifacts"][0]["id"]
        variants.append(duplicate_id)
        duplicate_path = copy.deepcopy(original)
        duplicate_path["artifacts"][1]["path"] = duplicate_path["artifacts"][0]["path"]
        duplicate_path["artifacts"][1]["sha256"] = duplicate_path["artifacts"][0]["sha256"]
        variants.append(duplicate_path)
        for variant in variants:
            with self.subTest(variant=variant):
                self.candidate.write_text(json.dumps(variant))
                with self.assertRaises(quality.QualityError):
                    quality.load_manifest(self.candidate)
        self.candidate.write_text(json.dumps(original))

    def test_changed_protocol_and_missing_artifact_sets_never_pass(self):
        self.candidate = self.make_manifest(self.root / "other", {"model": "LTX-2.5", "frames": 4})
        code, report = self.invoke()
        self.assertNotEqual(code, 0)
        self.assertIn("protocol digest mismatch", report["comparisons"][0]["errors"][0])
        value = json.loads(self.candidate.read_text(encoding="utf-8"))
        value["protocol"] = json.loads(self.reference.read_text(encoding="utf-8"))["protocol"]
        value["protocol_sha256"] = quality.protocol_digest(value["protocol"])
        value["artifacts"][0]["id"] = "unexpected-prompt"
        self.candidate.write_text(json.dumps(value))
        code, report = self.invoke()
        self.assertNotEqual(code, 0)
        self.assertIn("artifact ID set mismatch", report["comparisons"][0]["errors"][0])

    def test_metadata_mismatches_and_missing_streams_fail(self):
        reference = quality.parse_probe(media_metadata())
        for field, value in (("width", 1280), ("frames", 2), ("fps", "25"), ("channels", 1),
                             ("audio_duration", 0.12), ("video_start", 0.1)):
            with self.subTest(field=field), self.assertRaises(quality.QualityError):
                quality.compare_metadata(reference, {**reference, field: value})
        metadata = media_metadata()
        metadata["streams"].pop()
        with self.assertRaises(quality.QualityError):
            quality.parse_probe(metadata)

    def test_both_outputs_with_wrong_shape_or_audio_format_fail_frozen_protocol(self):
        for field, value in (("width", 1280), ("nb_read_frames", "2"), ("avg_frame_rate", "25/1"),
                             ("channels", 1), ("sample_rate", "44100")):
            def tool(argv, evidence, label, timeout):
                if label.endswith("probe"):
                    metadata = media_metadata()
                    stream = metadata["streams"][1 if field in {"channels", "sample_rate"} else 0]
                    stream[field] = value
                    return subprocess.CompletedProcess(argv, 0, json.dumps(metadata).encode(), b"")
                return self.fake_tool(argv, evidence, label, timeout)
            with self.subTest(field=field):
                code, report = self.invoke(fake=tool)
                self.assertNotEqual(code, 0)
                self.assertIn("frozen protocol", report["comparisons"][0]["pairs"][0]["error"])

    def test_frame_evidence_must_have_exact_count_order_and_finite_ssim(self):
        path = self.root / "ssim.log"
        for text in ("n:1 All:1\nn:2 All:1\n", "n:2 All:1\nn:1 All:1\nn:3 All:1\n",
                     "n:1 All:nan\nn:2 All:1\nn:3 All:1\n", "n:1 All:inf\nn:2 All:1\nn:3 All:1\n"):
            with self.subTest(text=text), self.assertRaises(quality.QualityError):
                path.write_text(text)
                quality.video_stats(path, "All", 3)

    def test_reference_and_candidate_with_same_short_audio_or_wrong_duration_fail(self):
        for stream_index, field, value in ((1, "duration", "0.001"), (0, "duration", "1.0"),
                                            (1, "start_time", "0.2"), (0, "start_time", "0.2")):
            def tool(argv, evidence, label, timeout):
                if label.endswith("probe"):
                    metadata = media_metadata()
                    metadata["streams"][stream_index][field] = value
                    return subprocess.CompletedProcess(argv, 0, json.dumps(metadata).encode(), b"")
                return self.fake_tool(argv, evidence, label, timeout)
            with self.subTest(field=field, value=value):
                code, report = self.invoke(fake=tool)
                self.assertNotEqual(code, 0)
                self.assertIn("frozen protocol", report["comparisons"][0]["pairs"][0]["error"])

    def test_decoded_audio_must_cover_clip_even_if_both_headers_match(self):
        def tool(argv, evidence, label, timeout):
            result = self.fake_tool(argv, evidence, label, timeout)
            if label.endswith("audio"):
                Path(argv[-1]).write_bytes(array("f", [0.1, 0.2] * 2).tobytes())
            return result
        code, report = self.invoke(fake=tool)
        self.assertNotEqual(code, 0)
        self.assertIn("decoded audio duration", report["comparisons"][0]["pairs"][0]["error"])

    def test_semantic_or_helper_drift_and_validator_drift_fail(self):
        original = json.loads(self.candidate.read_text(encoding="utf-8"))
        for field, changed in (("helper_sha256", "c" * 64),
                               ("semantic_sources_hashes", {"ltx2_recipes.py": "d" * 64})):
            value = copy.deepcopy(original)
            value[field] = changed
            self.candidate.write_text(json.dumps(value), encoding="utf-8")
            code, report = self.invoke()
            self.assertNotEqual(code, 0)
            self.assertIn("mismatch", report["comparisons"][0]["errors"][0])
        self.candidate.write_text(json.dumps(original), encoding="utf-8")
        code, report = self.invoke(["--expected-validator-sha", "f" * 64])
        self.assertNotEqual(code, 0)
        self.assertIn("validator changed", report["errors"][0])

    def test_incomplete_manifest_cannot_be_quality_reference(self):
        value = json.loads(self.reference.read_text(encoding="utf-8"))
        value["status"] = "failed"
        self.reference.write_text(json.dumps(value), encoding="utf-8")
        code, report = self.invoke()
        self.assertNotEqual(code, 0)
        self.assertIn("completed non-diagnostic", report["errors"][0])

    def test_pcm_requires_matching_stereo_length_finite_samples_and_silent_protection(self):
        self.assertEqual(quality.audio_nrmse([0, 0], [0, 0])["nrmse"], 0)
        for reference, candidate in (([], []), ([0.1, 0.2], [0.1]), ([0.1], [0.1]),
                                     ([0, 0], [0, 1e-9]), ([math.nan, 0], [0, 0]), ([0.1, 0.2], [math.inf, 0])):
            with self.subTest(candidate=candidate), self.assertRaises(quality.QualityError):
                quality.audio_nrmse(reference, candidate)

    def test_quiet_channel_regression_cannot_be_diluted_by_loud_channel(self):
        result = quality.audio_nrmse([1.0, 0.001] * 10, [1.0, 0.0] * 10)
        self.assertEqual(result["nrmse"], 1.0)
        self.assertEqual([channel["nrmse"] for channel in result["channels"]], [0.0, 1.0])
        with self.assertRaisesRegex(quality.QualityError, "channel 1"):
            quality.audio_nrmse([1.0, 0.0] * 10, [1.0, 0.00001] * 10)

    def test_directory_mode_rejects_duplicate_names_in_recursive_trials(self):
        directory = self.root / "repeated"
        for subdirectory in ("trial-001", "trial-002"):
            child = directory / subdirectory
            child.mkdir(parents=True)
            (child / "prompt-000-seed-42.mp4").write_bytes(b"fake")
        with self.assertRaisesRegex(quality.QualityError, "duplicate artifact basename"):
            quality.load_directory(directory)

    def make_run(self):
        run = self.root / "run"
        run.mkdir()
        (run / "run.lock.json").write_text(json.dumps({"repetitions": 2}))
        for side in ("baseline", "candidate"):
            for index in range(1, 3):
                trial = run / side / f"trial-{index:03d}"
                self.make_manifest(trial)
                (trial / "execution.json").write_text(json.dumps({"status": "completed", "returncode": 0}))
        return run

    def test_run_mode_checks_all_baseline_and_candidate_trials_independently(self):
        run = self.make_run()
        with patch.object(quality, "_run", side_effect=self.fake_tool), redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--candidate-root", str(run),
                                 "--output", str(self.output)])
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual([record["trial"] for record in report["comparisons"]],
                         ["baseline/trial-001", "baseline/trial-002", "candidate/trial-001", "candidate/trial-002"])
        self.assertEqual(len(self.video_calls), 12)
        self.assertEqual(len({argv[argv.index("-filter_complex") - 1] for argv in self.video_calls}), 12)

    def test_omnirsi_validation_uses_environment_run_root_and_all_trials(self):
        run = self.make_run()
        validation = run / "validation" / "result.json"
        validation.parent.mkdir()
        with patch.dict(os.environ, {"OMNIRSI_RESULT_PATH": str(validation)}), \
             patch.object(quality, "_run", side_effect=self.fake_tool), redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--omnirsi-run",
                                 "--output", str(validation)])
        report = json.loads(validation.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(len(report["comparisons"]), 4)
        self.assertEqual(report["validator_sha256"], quality.sha256_file(SCRIPT))

    def test_dependency_change_is_not_same_environment_performance(self):
        reference = json.loads(self.reference.read_text(encoding="utf-8"))
        candidate = json.loads(self.candidate.read_text(encoding="utf-8"))
        reference["dependency_versions"] = {"torch": "baseline", "vllm-omni": "old-target"}
        candidate["dependency_versions"] = {"torch": "changed", "vllm-omni": "new-target"}
        self.reference.write_text(json.dumps(reference), encoding="utf-8")
        self.candidate.write_text(json.dumps(candidate), encoding="utf-8")
        code, report = self.invoke()
        self.assertNotEqual(code, 0)
        self.assertIn("dependency versions mismatch", report["comparisons"][0]["errors"][0])
        candidate["dependency_versions"]["torch"] = "baseline"
        self.candidate.write_text(json.dumps(candidate), encoding="utf-8")
        code, _ = self.invoke()
        self.assertEqual(code, 0)

    def test_final_trials_cannot_change_the_frozen_parallel_winner(self):
        code, _ = self.invoke(["--expected-parallel", "1,4,4"])
        self.assertEqual(code, 0)
        candidate = json.loads(self.candidate.read_text(encoding="utf-8"))
        candidate["parallel"].update(tp=2, ulysses=2)
        self.candidate.write_text(json.dumps(candidate), encoding="utf-8")
        code, report = self.invoke(["--expected-parallel", "1,4,4"])
        self.assertNotEqual(code, 0)
        self.assertIn("frozen winner", report["comparisons"][0]["errors"][0])

    def test_run_mode_rejects_incomplete_execution_or_trial_coverage(self):
        run = self.make_run()
        (run / "baseline/trial-002/execution.json").write_text(json.dumps({"status": "failed", "returncode": 1}))
        with self.assertRaisesRegex(quality.QualityError, "did not complete"):
            quality.run_manifests(run)
        (run / "baseline/trial-002/execution.json").write_text(json.dumps({"status": "completed", "returncode": 0}))
        shutil.rmtree(run / "candidate/trial-002")
        with self.assertRaisesRegex(quality.QualityError, "coverage mismatch"):
            quality.run_manifests(run)

    def test_run_mode_does_not_allow_reusing_another_trials_files(self):
        run = self.make_run()
        source = json.loads((run / "candidate/trial-001/result.json").read_text(encoding="utf-8"))
        (run / "candidate/trial-002/result.json").write_text(json.dumps(source))
        with patch.object(quality, "_run", side_effect=self.fake_tool), redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--candidate-root", str(run),
                                 "--output", str(self.output)])
        self.assertNotEqual(code, 0)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertIn("escapes its trial directory", report["comparisons"][-1]["errors"][0])

    def test_real_cli_missing_tools_fails_and_preserves_json(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--reference-manifest", str(self.reference),
                                 "--candidate-manifest", str(self.candidate), "--output", str(self.output),
                                 "--ffprobe", "omnirsi-missing-ffprobe-executable"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))["verdict"], "FAIL")

    def test_nonfinite_threshold_and_tool_error_do_not_pass(self):
        code, report = self.invoke(["--min-ssim", "nan"])
        self.assertNotEqual(code, 0)
        self.assertIn("must be finite", report["errors"][0])
        with patch.object(quality, "_run", side_effect=quality.QualityError("ffmpeg exited 1")), redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--candidate-manifest", str(self.candidate),
                                 "--output", str(self.output)])
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))["verdict"], "FAIL")

    def test_output_cannot_overwrite_frozen_reference_manifest(self):
        before = self.reference.read_bytes()
        with redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--candidate-manifest", str(self.candidate),
                                 "--output", str(self.reference)])
        self.assertNotEqual(code, 0)
        self.assertEqual(self.reference.read_bytes(), before)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe are not installed locally")
    def test_real_ffmpeg_identical_short_av_clips(self):
        for manifest in (self.reference, self.candidate):
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["protocol"]["width"] = 64
            value["protocol"]["height"] = 64
            value["protocol_sha256"] = quality.protocol_digest(value["protocol"])
            for entry in value["artifacts"]:
                command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                           "testsrc2=size=64x64:rate=24:duration=0.125", "-f", "lavfi", "-i",
                           "sine=frequency=440:sample_rate=48000:duration=0.125", "-c:v", "mpeg4",
                           "-c:a", "aac", "-ac", "2", "-shortest", entry["path"]]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                entry["sha256"] = quality.sha256_file(entry["path"])
            manifest.write_text(json.dumps(value))
        with redirect_stdout(io.StringIO()):
            code = quality.main(["--reference-manifest", str(self.reference), "--candidate-manifest", str(self.candidate),
                                 "--output", str(self.output)])
        self.assertEqual(code, 0, self.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
