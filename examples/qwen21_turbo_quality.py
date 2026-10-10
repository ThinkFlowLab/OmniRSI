"""Independent PNG accuracy and frozen-protocol gate for Qwen Turbo trials."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import ltx25_b300 as lifecycle


def load(path):
    value=json.loads(Path(path).read_text())
    if value.get('status')!='completed' or value.get('diagnostic') or not value.get('source_integrity',{}).get('verified'):
        raise ValueError('Completed non-diagnostic source-stable trial required')
    if lifecycle.digest(value['protocol'])!=value['protocol_sha256']:
        raise ValueError('Invalid protocol digest')
    return value


def validate(reference_path,candidate_path,expected_parallel=None):
    from PIL import Image
    import numpy as np
    from skimage.metrics import structural_similarity
    ref,cand=load(reference_path),load(candidate_path)
    for name in ['protocol','protocol_sha256','protected_sources','guard_hashes','dependency_versions','model_assets_sha256']:
        if ref[name]!=cand[name]:raise ValueError(f'Frozen {name} mismatch')
    if expected_parallel and (cand['parallel']['tp'],cand['parallel']['ulysses'])!=expected_parallel:
        raise ValueError('Frozen winner parallel mismatch')
    refs={item['id']:item for item in ref['artifacts'] if item['group']==1}
    expected={item['id'] for item in ref['protocol']['prompts']}
    if set(refs)!=expected or len(ref['artifacts'])!=3 or len(ref['groups_ms'])!=1:
        raise ValueError('Canonical reference coverage mismatch')
    groups={item['group'] for item in cand['artifacts']}
    if groups!=set(range(1,len(cand['groups_ms'])+1)):raise ValueError('Missing candidate group')
    if len(cand['artifacts'])!=len(groups)*len(expected):raise ValueError('Wrong artifact count')
    results=[]
    seen=set()
    pairs = [(item, refs.get(item['id'])) for item in cand['artifacts']]
    if cand.get('first_request',{}).get('seed') != ref.get('first_request',{}).get('seed'):
        raise ValueError('First-request seed mismatch')
    pairs.append(({'group':0,'id':'first-request',**cand['first_request']},ref['first_request']))
    for item,other in pairs:
        identifier=(item['group'],item['id'])
        if identifier in seen or other is None:raise ValueError('Duplicate or unexpected output')
        seen.add(identifier)
        arrays=[]
        for entry in [other,item]:
            if lifecycle.sha256(entry['path'])!=entry['sha256']:raise ValueError('PNG hash mismatch')
            with Image.open(entry['path']) as image:
                image.load()
                if image.format!='PNG' or image.size!=(1280,704) or image.mode!='RGBA':
                    raise ValueError('Expected complete 1280x704 native RGBA PNG')
                arrays.append(np.asarray(image).copy())
        a,b=arrays
        mse=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
        psnr=math.inf if mse==0 else 10*math.log10(255**2/mse)
        ssim=float(structural_similarity(a,b,channel_axis=2,data_range=255))
        channel_metrics=[]
        for name, left, right in [('rgb',a[:,:,:3],b[:,:,:3]),('alpha',a[:,:,3],b[:,:,3])]:
            channel_mse=float(np.mean((left.astype(np.float64)-right.astype(np.float64))**2))
            channel_psnr=math.inf if channel_mse==0 else 10*math.log10(255**2/channel_mse)
            channel_ssim=float(structural_similarity(left,right,channel_axis=2 if name=='rgb' else None,data_range=255))
            channel_metrics.append({'channel':name,'ssim':channel_ssim,'psnr_db':'inf' if channel_mse==0 else channel_psnr,
                                    'passed':channel_ssim>=0.99 and channel_psnr>=35})
        passed=all(x['passed'] for x in channel_metrics)
        results.append({'group':item['group'],'id':item['id'],'ssim':ssim,'psnr_db':'inf' if mse==0 else psnr,
                        'channels':channel_metrics,'verdict':'PASS' if passed else 'FAIL'})
    return {'verdict':'PASS' if all(x['verdict']=='PASS' for x in results) else 'FAIL','comparisons':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-manifest',required=True)
    p.add_argument('--candidate-manifest')
    p.add_argument('--omnirsi-run',action='store_true')
    p.add_argument('--expected-parallel')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        parallel=tuple(map(int,args.expected_parallel.split(','))) if args.expected_parallel else None
        if args.omnirsi_run:
            root=Path(os.environ['OMNIRSI_RESULT_PATH']).parent
            repetitions=json.loads((root/'run.lock.json').read_text())['repetitions']
            paths=[]
            for side in ['baseline','candidate']:
                for index in range(1,repetitions+1):
                    trial=root/side/f'trial-{index:03d}'
                    execution=json.loads((trial/'execution.json').read_text())
                    if execution['status']!='completed' or execution['returncode']!=0:raise ValueError('Incomplete trial')
                    paths.append(trial/'result.json')
        elif args.candidate_manifest:paths=[Path(args.candidate_manifest)]
        else:raise ValueError('Candidate manifest or complete OmniRSI run required')
        reports=[{'manifest':str(path),**validate(args.reference_manifest,path,parallel)} for path in paths]
        result={'verdict':'PASS' if all(r['verdict']=='PASS' for r in reports) else 'FAIL','min_ssim':0.99,'min_psnr_db':35,'reports':reports}
    except Exception as error:result={'verdict':'FAIL','error':str(error)}
    lifecycle.write_json(args.output,result)
    print(json.dumps({'verdict':result['verdict'],'output':args.output}),flush=True)
    return 0 if result['verdict']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
