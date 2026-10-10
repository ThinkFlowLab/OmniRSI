"""CPU contracts and Linux-only process ownership regressions for the campaign."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from omnirsi.execution import _execute, repository_identity


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ltx25_b300.py"
SPEC = importlib.util.spec_from_file_location("ltx25_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def live_process(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "source"
        self.repo.mkdir()
        (self.repo / "semantics.py").write_text("fixed recipe\n", encoding="utf-8")
        (self.repo / "kernel.py").write_text("original kernel\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.git("-c", "user.name=OmniRSI test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                 "commit", "--quiet", "-s", "-m", "fixture")
        self.identity = repository_identity(self.repo)
        self.semantic_patch = patch.object(campaign, "SEMANTIC_SOURCE_PATHS", ("semantics.py",))
        self.semantic_patch.start()
        self.addCleanup(self.semantic_patch.stop)
        self.version_patch = patch.object(campaign, "dependency_versions", return_value={"torch": "test", "diffusers": "test"})
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)
        self.quality_script = self.root / "quality.py"
        self.quality_script.write_text("# fake quality command used only by CPU contracts\n", encoding="utf-8")
        self.reference = self.make_reference()

    def git(self, *argv):
        return subprocess.run(["git", "-C", str(self.repo), *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    def make_reference(self):
        directory = self.root / "reference"
        directory.mkdir()
        artifacts = []
        for item in campaign.protocol()["prompts"]:
            artifact = directory / (item["id"] + ".mp4")
            artifact.write_bytes(b"CPU fixture, not a generated video")
            artifacts.append({"id": item["id"], "path": str(artifact), "sha256": campaign.sha256(artifact)})
        path = directory / "result.json"
        campaign.write_json(path, {"status": "completed", "source_commit": self.identity["head"],
                                  "protocol": campaign.protocol(), "protocol_sha256": campaign.digest(campaign.protocol()),
                                  "semantic_sources_hashes": campaign.semantic_sources(self.repo),
                                  "helper_sha256": campaign.sha256(SCRIPT),
                                  "dependency_versions": campaign.dependency_versions(), "artifacts": artifacts})
        return path

    def search_args(self, extra=()):
        return campaign.parser().parse_args(["search", "--source-repo", str(self.repo), "--cache-root", str(self.root / "cache"),
            "--reference-manifest", str(self.reference), "--quality-script", str(self.quality_script),
            "--output-dir", str(self.root / "search"), "--repetitions", "1", *extra])

    def trial_args(self, extra=()):
        return campaign.parser().parse_args(["trial", "--source-repo", str(self.repo), "--cache-root", str(self.root / "cache"),
            "--output", str(self.root / "trial" / "result.json"), *extra])

    def test_parallel_matrix_and_single_reference_constraints(self):
        for tp, ulysses in ((1, 4), (2, 2), (4, 1)):
            for vae in (1, 2, 4):
                self.assertEqual(campaign.validate_parallel(tp, ulysses, vae), 4)
        self.assertEqual(campaign.validate_parallel(1, 1, 1, True), 1)
        for args in ((2, 4, 1, False), (4, 4, 4, False), (1, 4, 8, False), (1, 4, 0, False),
                     (1, 1, 2, True), (2, 1, 1, True), (1, 2, 1, False), (True, 4, 1, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                campaign.validate_parallel(*args)

    def test_protocol_freezes_warmup_seeds_sigmas_guidance_decoder_codec(self):
        value = campaign.protocol()
        measured = {item["seed"] for item in value["prompts"]}
        self.assertEqual(measured, {42, 43, 44})
        self.assertEqual(value["warmup_seeds"], [[10042, 10043, 10044], [11042, 11043, 11044]])
        self.assertFalse(measured.intersection(seed for round_seeds in value["warmup_seeds"] for seed in round_seeds))
        self.assertEqual((len(value["stage_1_sigmas"]), len(value["stage_2_sigmas"])), (9, 4))
        self.assertEqual(value["guidance"], "positive_only")
        self.assertFalse(value["ltx2_use_conv_vae"])
        self.assertEqual(value["video_codec_options"], {"preset": "ultrafast", "threads": "0", "crf": "18"})
        argv = campaign.server_argv(self.trial_args())
        self.assertEqual(json.loads(argv[argv.index("--stage-overrides") + 1]), {"0": {"extras": {"ltx2_use_conv_vae": False}}})
        self.assertIn("--cfg-parallel-size", argv)
        self.assertEqual(argv[argv.index("--ring") + 1], "1")

    def test_request_sends_frozen_sigmas_and_codec_as_extra_params(self):
        response = Mock(status=200, headers={"Content-Type": "video/mp4"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.side_effect = [b"CPU fixture MP4 bytes", b""]
        with patch.object(campaign.urllib.request, "urlopen", return_value=response) as urlopen:
            value = campaign.request_video(self.trial_args(), campaign.protocol()["prompts"][0], self.root / "request.mp4")
        self.assertGreater(value, 0)
        payload = urlopen.call_args.args[0].data.decode("utf-8")
        self.assertIn('name="extra_params"', payload)
        self.assertIn('"stage_1_sigmas":', payload)
        self.assertIn('"stage_2_sigmas":', payload)
        self.assertIn('"crf": "18"', payload)

    def test_dependency_policy_preserves_runtime_versions_and_allows_omni_version_text(self):
        before = {"vllm-omni": "dev.old", "diffusers": "1.0", "torch": "2.0"}
        after = {**before, "vllm-omni": "dev.new"}
        self.assertEqual(campaign.semantic_dependency_versions(before), campaign.semantic_dependency_versions(after))
        after["diffusers"] = "2.0"
        self.assertNotEqual(campaign.semantic_dependency_versions(before), campaign.semantic_dependency_versions(after))

    def test_dry_run_is_cross_platform_and_never_calls_gpu_or_subprocess(self):
        args = ["search", "--source-repo", str(self.root / "not-installed"), "--cache-root", str(self.root / "cache"),
                "--reference-manifest", "unread-reference.json", "--quality-script", "unread-guard.py",
                "--output-dir", str(self.root / "dry-search"), "--include-tp-candidates", "--dry-run"]
        stdout = io.StringIO()
        with patch.object(campaign, "preflight", side_effect=AssertionError("GPU inspection must not execute")), \
             patch.object(campaign.subprocess, "run", side_effect=AssertionError("no child processes in dry run")), \
             redirect_stdout(stdout):
            self.assertEqual(campaign.main(args), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())["grid"]), 9)
        self.assertFalse((self.root / "dry-search").exists())

    def test_tp_candidates_need_three_repeats_before_any_allocation(self):
        args = self.search_args(("--include-tp-candidates", "--repetitions", "2"))
        with self.assertRaisesRegex(ValueError, "at least three"):
            campaign.search(args)
        self.assertFalse(Path(args.output_dir).exists())

    def test_storage_under_source_is_rejected_before_allocation(self):
        args = self.trial_args()
        for cache, output in ((self.repo / "cache", self.root / "new-output"),
                              (self.root / "new-cache", self.repo / "runs")):
            args.cache_root = str(cache)
            with self.subTest(cache=cache), self.assertRaisesRegex(ValueError, "outside"):
                campaign.storage_locations(args, output)
            self.assertFalse(cache.exists())
            self.assertFalse(output.exists())
        with patch.dict(os.environ, {"HF_HOME": str(self.repo / "hf-cache")}):
            args.cache_root = str(self.root / "cache")
            with self.assertRaisesRegex(ValueError, "HF_HOME"):
                campaign.storage_locations(args, self.root / "trial")

    def test_search_rejects_dirty_source_and_tampered_reference(self):
        args = self.search_args()
        (self.repo / "kernel.py").write_text("local change\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean pinned"):
            campaign.freeze_campaign(args)
        (self.repo / "kernel.py").write_text("original kernel\n", encoding="utf-8")
        value = json.loads(self.reference.read_text(encoding="utf-8"))
        Path(value["artifacts"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            campaign.freeze_campaign(args)

    def test_frozen_evidence_detects_guard_source_and_reference_drift(self):
        args = self.search_args()
        frozen = campaign.freeze_campaign(args)
        campaign.assert_campaign_frozen(args, frozen)
        self.quality_script.write_text("changed guard\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignDrift, "Quality guard"):
            campaign.assert_campaign_frozen(args, frozen)
        self.quality_script.write_text("# fake quality command used only by CPU contracts\n", encoding="utf-8")
        (self.repo / "kernel.py").write_text("changed source\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignDrift, "Source changed"):
            campaign.assert_campaign_frozen(args, frozen)

    def simulated_search(self, args, quality_pass=True, drift=False):
        original_run = subprocess.run
        changed = [False]
        calls = []
        def identity(_):
            return {**self.identity, "snapshot_hash": "changed" if changed[0] else self.identity["snapshot_hash"], "dirty": changed[0]}
        def trial(trial_args):
            calls.append((trial_args.tp, trial_args.ulysses, trial_args.vae))
            result = {"status": "completed", "source_integrity": {"verified": True}, "server_cleanup": {"remaining_pids": []},
                      "metrics": {"latency_ms": 1000 / trial_args.vae}}
            campaign.write_json(trial_args.output, result)
            if drift:
                changed[0] = True
            return result
        def subprocess_run(argv, *positional, **keywords):
            if len(argv) > 1 and argv[1] == str(self.quality_script):
                campaign.write_json(argv[argv.index("--output") + 1], {"verdict": "PASS" if quality_pass else "FAIL"})
                return subprocess.CompletedProcess(argv, 0 if quality_pass else 1)
            return original_run(argv, *positional, **keywords)
        with patch.object(campaign, "source_snapshot", side_effect=identity), \
             patch.object(campaign, "run_trial", side_effect=trial), \
             patch.object(campaign.subprocess, "run", side_effect=subprocess_run), redirect_stdout(io.StringIO()):
            campaign.search(args)
        return calls

    def test_search_selects_only_tested_quality_passing_cells_and_records_hashes(self):
        args = self.search_args()
        calls = self.simulated_search(args)
        self.assertEqual(calls, [(1, 4, 1), (1, 4, 2), (1, 4, 4)])
        best = json.loads((Path(args.output_dir) / "best.json").read_text(encoding="utf-8"))
        self.assertEqual(best["best"]["vae"], 4)
        self.assertIn("not a global", best["scope"])
        self.assertEqual(best["frozen_evidence"]["quality_script"]["sha256"], campaign.sha256(self.quality_script))
        self.assertEqual(best["frozen_evidence"]["reference_manifest"]["sha256"], campaign.sha256(self.reference))

    def test_explicit_tp_grid_qualifies_only_after_three_quality_passing_trials(self):
        args = self.search_args(("--include-tp-candidates", "--repetitions", "3"))
        calls = self.simulated_search(args)
        self.assertEqual(len(calls), 27)
        report = json.loads((Path(args.output_dir) / "search.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["cells"]), 9)
        for cell in report["cells"]:
            self.assertEqual(cell["status"], "eligible")
            self.assertEqual(len(cell["trials"]), 3)
            if cell["tp"] != 1:
                self.assertEqual(cell["tp_qualification"], "three_or_more_quality_passing_repeats")

    def test_quality_failure_and_source_drift_never_create_best(self):
        args = self.search_args()
        with self.assertRaisesRegex(ValueError, "No quality-passing"):
            self.simulated_search(args, quality_pass=False)
        self.assertFalse((Path(args.output_dir) / "best.json").exists())
        args.output_dir = str(self.root / "drift-search")
        with self.assertRaises(campaign.CampaignDrift):
            self.simulated_search(args, drift=True)
        record = json.loads((Path(args.output_dir) / "search.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "aborted")
        self.assertFalse((Path(args.output_dir) / "best.json").exists())

    def simulated_trial(self, mutate=False, dirty=False):
        if dirty:
            (self.repo / "kernel.py").write_text("fixed dirty candidate\n", encoding="utf-8")
        args = self.trial_args()
        args.vae = 1
        process = Mock(pid=999, returncode=0)
        process.poll.return_value = None
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        seeds = []
        def request(_, item, destination):
            seeds.append(item["seed"])
            Path(destination).write_bytes(b"CPU fixture, not a generated video")
            if mutate and len(seeds) == 9:
                (self.repo / "kernel.py").write_text("changed during request\n", encoding="utf-8")
            return 1000.0
        with patch.object(campaign.sys, "platform", "linux"), patch.object(campaign, "preflight", return_value=[]), \
             patch.object(campaign, "server_argv", return_value=[sys.executable]), \
             patch.object(campaign, "launch_server", return_value=(process, {"mode": "CPU mock"})), \
             patch.object(campaign, "stop_server", return_value={"remaining_pids": []}), \
             patch.object(campaign.urllib.request, "urlopen", return_value=response), \
             patch.object(campaign, "request_video", side_effect=request):
            result = campaign.run_trial(args)
        return result, seeds

    def test_trial_records_stable_dirty_candidate_and_distinct_warmup_seeds(self):
        result, seeds = self.simulated_trial(dirty=True)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["source_integrity"]["verified"])
        self.assertTrue(result["source_integrity"]["dirty_at_start"])
        self.assertEqual(seeds, [10042, 10043, 10044, 11042, 11043, 11044, 42, 43, 44])

    def test_trial_source_mutation_preserves_failed_manifest(self):
        result, _ = self.simulated_trial(mutate=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["source_integrity"]["verified"])
        self.assertTrue((self.root / "trial/result.json").is_file())

    def test_generated_trial_manifest_satisfies_quality_frozen_parallel_contract(self):
        result, _ = self.simulated_trial()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["parallel"], {"tp": 1, "ulysses": 4, "vae": 1, "cfg": 1, "ring": 1})
        quality_path = SCRIPT.with_name("ltx25_quality.py")
        quality_spec = importlib.util.spec_from_file_location("campaign_quality_contract", quality_path)
        quality = importlib.util.module_from_spec(quality_spec)
        quality_spec.loader.exec_module(quality)
        manifest = self.root / "trial/result.json"
        arguments = ["--reference-manifest", str(self.reference), "--candidate-manifest", str(manifest),
                     "--expected-parallel", "1,4,1", "--output", str(self.root / "quality.json")]
        # Exercise real driver serialization and guard manifest/configuration validation;
        # only media decoding is replaced, since these CPU artifacts are not generated videos.
        with patch.object(quality, "compare_pair", return_value={"verdict": "PASS"}) as decode, redirect_stdout(io.StringIO()):
            self.assertEqual(quality.main(arguments), 0)
            self.assertEqual(decode.call_count, 3)
        del result["parallel"]["ring"]
        campaign.write_json(manifest, result)
        with patch.object(quality, "compare_pair", return_value={"verdict": "PASS"}) as decode, redirect_stdout(io.StringIO()):
            self.assertNotEqual(quality.main(arguments), 0)
            decode.assert_not_called()


@unittest.skipUnless(sys.platform == "linux", "Process-group regressions require Linux; no GPU is used")
class LinuxLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.server = self.root / "fake_server.py"
        self.pids = self.root / "pids.json"
        self.server.write_text(
            "import json,os,pathlib,signal,subprocess,sys,time\n"
            "ready=sys.argv[1]+'.worker-ready'\n"
            "worker=subprocess.Popen([sys.executable,'-c','import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(\"ready\"); time.sleep(120)',ready])\n"
            "while not pathlib.Path(ready).exists(): time.sleep(.01)\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'server':os.getpid(),'worker':worker.pid,'group':os.getpgrp()}))\n"
            "time.sleep(.1)\n"
            "if '--exit-leader' not in sys.argv: time.sleep(120)\n", encoding="utf-8")
        self.loader = ("import importlib.util,json,os,pathlib,sys,time\n"
                       f"spec=importlib.util.spec_from_file_location('campaign',{str(SCRIPT)!r})\n"
                       "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n")

    def test_runner_timeout_kills_inherited_server_and_worker_even_without_helper_finally(self):
        helper = self.root / "helper.py"
        helper.write_text(self.loader +
            "log=open(sys.argv[3],'wb')\n"
            "process,ownership=module.launch_server([sys.executable,sys.argv[1],sys.argv[2]], pathlib.Path(sys.argv[1]).parent, os.environ.copy(),log)\n"
            "pathlib.Path(sys.argv[4]).write_text(json.dumps(ownership))\n"
            "time.sleep(120)\n", encoding="utf-8")
        owner = self.root / "ownership.json"
        result = _execute((sys.executable, str(helper), str(self.server), str(self.pids), str(self.root / "server.log"), str(owner)),
                          self.root, self.root / "trial", time.monotonic() + 5, None)
        self.assertEqual(result["status"], "timeout")
        ownership = json.loads(owner.read_text())
        pids = json.loads(self.pids.read_text())
        self.assertEqual(ownership["mode"], "omnirsi_trial_group")
        self.assertFalse(ownership["server_start_new_session"])
        self.assertEqual(pids["group"], ownership["helper_pid"])
        for key in ("server", "worker"):
            self.assertFalse(live_process(pids[key]), f"{key} survived helper timeout")

    def test_direct_cleanup_kills_workers_after_server_leader_already_exited(self):
        with patch.dict(os.environ, {}, clear=False):
            old = os.environ.pop("OMNIRSI_RESULT_PATH", None)
            try:
                with (self.root / "server.log").open("wb") as log:
                    process, owner = campaign.launch_server([sys.executable, str(self.server), str(self.pids), "--exit-leader"],
                                                           self.root, os.environ.copy(), log)
                    self.addCleanup(campaign.stop_server, process, owner, 0.1)
                    process.wait(timeout=10)
                    pids = json.loads(self.pids.read_text())
                    self.assertTrue(live_process(pids["worker"]))
                    cleanup = campaign.stop_server(process, owner, grace_seconds=0.1)
            finally:
                if old is not None:
                    os.environ["OMNIRSI_RESULT_PATH"] = old
        self.assertEqual(owner["mode"], "server_session")
        self.assertNotEqual(owner["process_group_id"], os.getpgrp())
        self.assertTrue(cleanup["kill_sent"])
        self.assertEqual(cleanup["remaining_pids"], [])
        self.assertFalse(live_process(pids["worker"]))

    def test_normal_supervised_cleanup_preserves_helper_and_interactive_group(self):
        helper = self.root / "helper-stop.py"
        owner = self.root / "ownership.json"
        helper.write_text(self.loader +
            "log=open(sys.argv[3],'wb')\n"
            "process,ownership=module.launch_server([sys.executable,sys.argv[1],sys.argv[2]], pathlib.Path(sys.argv[1]).parent, os.environ.copy(),log)\n"
            "deadline=time.monotonic()+5\n"
            "while not pathlib.Path(sys.argv[2]).exists() and time.monotonic()<deadline: time.sleep(.02)\n"
            "time.sleep(.2)\n"
            "cleanup=module.stop_server(process,ownership,.1)\n"
            "pathlib.Path(sys.argv[4]).write_text(json.dumps({'ownership':ownership,'cleanup':cleanup,'helper_survived':True}))\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["OMNIRSI_RESULT_PATH"] = str(self.root / "result.json")
        original_group = os.getpgrp()
        process = subprocess.Popen([sys.executable, str(helper), str(self.server), str(self.pids), str(self.root / "server.log"), str(owner)],
                                   env=environment, start_new_session=True)
        try:
            returncode = process.wait(timeout=15)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        self.assertEqual(returncode, 0)
        self.assertEqual(os.getpgrp(), original_group)
        record = json.loads(owner.read_text())
        self.assertTrue(record["helper_survived"])
        self.assertEqual(record["cleanup"]["remaining_pids"], [])
        for pid in json.loads(self.pids.read_text()).values():
            self.assertFalse(live_process(pid))


if __name__ == "__main__":
    unittest.main()
