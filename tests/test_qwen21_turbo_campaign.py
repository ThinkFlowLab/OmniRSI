"""CPU contracts for the frozen Qwen Turbo sampling and image evidence gates."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

EXAMPLES = Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
import qwen21_turbo_b300 as campaign
import qwen21_turbo_quality as quality


class QwenTurboTests(unittest.TestCase):
    def test_run_validation_resolves_parent_of_validation_and_requires_all_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            root.mkdir()
            (root/'run.lock.json').write_text(json.dumps({'repetitions':1}))
            for side in ['baseline','candidate']:
                trial=root/side/'trial-001'
                trial.mkdir(parents=True)
                (trial/'execution.json').write_text(json.dumps({'status':'completed','returncode':0}))
            paths=quality.omnirsi_manifests(root/'validation/result.json')
            self.assertEqual(paths,[root/'baseline/trial-001/result.json',root/'candidate/trial-001/result.json'])
            (root/'candidate/trial-002').mkdir()
            with self.assertRaisesRegex(ValueError,'coverage'):
                quality.omnirsi_manifests(root/'validation/result.json')

    def test_precreated_runner_directory_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source'
            source.mkdir()
            model=root/campaign.REVISION
            model.mkdir()
            (model/'model_index.json').write_text(json.dumps({'sample_sigmas':campaign.SIGMAS}))
            output=root/'trial/result.json'
            output.parent.mkdir()
            args=SimpleNamespace(source_repo=str(source),output=str(output),cache_root=str(root/'cache'),
                                 tp=1,ulysses=4,single_reference=False,model_path=str(model),diagnostic=False,
                                 profile_steps=2,held_out=False,port=8098,startup_timeout=10,repetitions=1)
            with patch.object(campaign.lifecycle,'source_snapshot',return_value={'snapshot_hash':'fixed'}), \
                 patch.object(campaign,'protected_sources',return_value={'method':'fixed'}), \
                 patch.object(campaign,'guards',return_value={'guard':'fixed'}), \
                 patch.object(campaign,'dependencies',return_value={'version':'fixed'}), \
                 patch.object(campaign,'model_assets',return_value='fixed'), \
                 patch.object(campaign.lifecycle,'preflight',side_effect=ValueError('CPU probe failure')):
                result=campaign.run(args)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['error'],'CPU probe failure')
            self.assertTrue(output.is_file())
            self.assertTrue(result['source_integrity']['verified'])

    def test_native_schedule_and_diagnostic_are_distinct(self):
        full=campaign.protocol()
        short=campaign.protocol(True,2)
        self.assertEqual(len(full['sigmas']),8)
        self.assertEqual(short['sigmas'],[1.0,0.414568])
        self.assertNotEqual(campaign.lifecycle.digest(full),campaign.lifecycle.digest(short))
        self.assertEqual((full['width'],full['height']),(1280,704))
        self.assertEqual(full['true_cfg_scale'],1.0)
        with self.assertRaises(ValueError):campaign.protocol(True,0)

    def test_unseen_prompt_seed_set_does_not_overlap_main_or_warmups(self):
        main=campaign.protocol()
        held=campaign.protocol(held_out=True)
        self.assertFalse({x['prompt'] for x in main['prompts']}&{x['prompt'] for x in held['prompts']})
        seeds={x['seed'] for x in held['prompts']}
        self.assertFalse(seeds&{x['seed'] for x in main['prompts']})
        self.assertFalse(seeds&{s for group in held['warmup_seeds'] for s in group})

    def test_http_request_overrides_sigmas_and_requires_complete_png(self):
        settings=campaign.protocol(True,2)
        response=Mock(status=200)
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps({'data':[{'b64_json':base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()}]}).encode()
        with tempfile.TemporaryDirectory() as temp, patch.object(campaign.urllib.request,'urlopen',return_value=response) as call:
            campaign.image_request(8098,settings['prompts'][0],Path(temp)/'image.png',settings)
            body=json.loads(call.call_args.args[0].data)
            self.assertEqual(body['sigmas'],[1.0,0.414568])
            self.assertNotIn('num_inference_steps',body)
            self.assertEqual(body['size'],'1280x704')
            response.read.return_value=b'{"data":[]}'
            with self.assertRaises(ValueError):campaign.image_request(8098,settings['prompts'][0],Path(temp)/'bad.png',settings)

    def test_diagnostic_failed_source_drift_and_digest_tampering_cannot_pass(self):
        good={'status':'completed','diagnostic':False,'source_integrity':{'verified':True},
              'protocol':campaign.protocol(),'protocol_sha256':campaign.lifecycle.digest(campaign.protocol())}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'result.json'
            for changes in [{'diagnostic':True},{'status':'failed'},{'source_integrity':{'verified':False}},{'protocol_sha256':'bad'}]:
                path.write_text(json.dumps({**good,**changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):quality.load(path)
            path.write_text(json.dumps(good))
            self.assertEqual(quality.load(path)['protocol'],good['protocol'])

    def test_asset_stat_or_file_coverage_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            model=root/campaign.REVISION
            model.mkdir()
            weight=model/'weight.bin'
            weight.write_bytes(b'original')
            (root/'artifacts').mkdir()
            st=weight.stat()
            manifest={'model':str(model),'revision':campaign.REVISION,'files':[{'path':'weight.bin','bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':campaign.lifecycle.sha256(weight)}]}
            (root/'artifacts/model-assets.json').write_text(json.dumps(manifest))
            original=campaign.model_assets(model,root/'cache')
            self.assertEqual(len(original),64)
            weight.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):campaign.model_assets(model,root/'cache')
            (model/'new.bin').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'coverage'):campaign.model_assets(model,root/'cache')

if __name__=='__main__':unittest.main()
