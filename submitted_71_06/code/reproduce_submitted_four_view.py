"""Reproduce the user's 71.0621387283237 submission using its fixed single model."""
import argparse,csv,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
MODEL_SHA='cd30a7d8afb6785d9d889dde65726315cc67b5e56c8ecc1a98ce1da59cc8d3fc'
TEST_SHA='73f02154ac07a947b225c071e0002da68511a7292fd86c0f48f2ea8cffe2aa495'
CSV_SHA='996e481af5b93d982cf483fc7cdf56ce177f2e17c7fa6c40ee874ac6ac7cd73d'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def export(features,names,policy,out):
    with np.load(features) as data:
        probabilities=data['probabilities'];indices=data['source_indices']
    assert probabilities.shape==(4,len(names),500) and np.array_equal(indices,np.arange(len(names)))
    p=np.maximum(probabilities,1e-9)**(1/policy['temperature']);p/=p.sum(-1,keepdims=True)
    posterior=np.einsum('v,vnc->nc',np.array(policy['view_weights']),p)
    labels=posterior.argmax(-1)
    with (out/'pred_results.csv').open('x',newline='') as stream:
        csv.writer(stream).writerows((n,f'{int(y):04d}') for n,y in zip(names,labels))
    with zipfile.ZipFile(out/'pred_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(out/'pred_results.csv','pred_results.csv')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=ROOT.parent/'model/model.pt')
    parser.add_argument('--output',type=Path,default=ROOT.parent/'reproduced_results')
    parser.add_argument('--batch',type=int,default=48)
    args=parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    assert digest(args.model)==MODEL_SHA,'Model differs from the submitted version'
    assert (ROOT/'test.zip').is_file(),'Place the original test.zip in code/test.zip'
    assert digest(ROOT/'test.zip')==TEST_SHA,'Test archive differs from the original 49857-image package'
    names=json.loads((ROOT/'data/test_names.json').read_text())['names']
    with zipfile.ZipFile(ROOT/'test.zip') as archive:
        assert sorted(n for n in archive.namelist() if not n.endswith('/'))==names
    assert len(names)==49857
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'pred_results.csv').exists(),'Choose a new output folder; existing result is preserved'
    (ROOT/'TEST_UPLOAD_VERIFIED').write_text(TEST_SHA+'\n')
    (ROOT/'image_cache').mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'build_current_test_cache.py')],cwd=ROOT,check=True)
    features=out/'features'
    if not (features/'test_binding.json').exists():
        subprocess.run([sys.executable,str(ROOT/'predict.py'),'--model',str(args.model.resolve()),'--output',str(features),'--new-checkpoint','--batch',str(args.batch)],cwd=ROOT,check=True)
    binding=json.loads((features/'test_binding.json').read_text())
    assert binding['checkpoint_sha256']==MODEL_SHA and binding['feature_cache_sha256']==digest(features/'test.npz')
    policy=json.loads((ROOT/'inference_policy.json').read_text())
    assert binding['actual_views']==policy['classification_views']
    export(features/'test.npz',names,policy,out)
    actual=digest(out/'pred_results.csv')
    report=dict(rows=len(names),checkpoint_sha256=MODEL_SHA,csv_sha256=actual,expected_csv_sha256=CSV_SHA,
        exact_original_csv=actual==CSV_SHA,method='single-model weighted four-view classification; no graph or class balancing')
    (out/'reproduction_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if actual!=CSV_SHA:raise RuntimeError('CSV differs from the submitted CSV; inspect the actual environment and retain the report')

if __name__=='__main__':main()
