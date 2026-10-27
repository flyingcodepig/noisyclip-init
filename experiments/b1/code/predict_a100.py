import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'engine'))
import torch
from infer_a100 import predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    checkpoint=a.model.resolve()
    cfg=torch.load(checkpoint,map_location='cpu',weights_only=False)['config']
    policy=json.loads((ROOT/'inference_policy.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'test.npz'
    if path.exists(): raise FileExistsError(path)
    predict(ROOT,checkpoint,path,cfg,48,'test',policy['classification_views'])
if __name__=='__main__': main()
