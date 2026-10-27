import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'engine'))
import torch
import numpy as np
from infer import predict, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-checkpoint',action='store_true',help='Use a newly trained checkpoint; retain file binding checks')
    p.add_argument('--batch',type=int,default=48)
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--indices',type=Path)
    a=p.parse_args()
    checkpoint=a.model or next(ROOT.parent.glob('*-模型文件.pt'))
    if not a.new_checkpoint:
        assert sha(checkpoint)=='9048b62c2ceff91743087c475d7a5f9f31d85131b44f7864c37418fbccba4348'
    cfg=torch.load(checkpoint,map_location='cpu',weights_only=False)['config']
    policy=json.loads((ROOT/'inference_policy.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'test.npz'
    if path.exists(): raise FileExistsError(path)
    indices=np.load(a.indices) if a.indices else None
    predict(ROOT,checkpoint,path,cfg,a.batch,a.split,policy['classification_views'],indices=indices)
    (a.output/'test_binding.json').write_text(json.dumps(dict(checkpoint_sha256=sha(checkpoint),feature_cache_sha256=sha(path),actual_views=policy['classification_views'],split=a.split,indices_sha256=sha(a.indices) if a.indices else None),indent=2))
if __name__=='__main__': main()
