"""Evaluate held-out TRAIN with the submitted four-view rule, never competition TEST."""
import argparse,csv,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn.attention import sdpa_kernel,SDPBackend
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'engine'))
from model_base import load_model
from raw_cache import RawArray

class Validation(Dataset):
    def __init__(self):
        self.center=RawArray(ROOT/'image_cache/val_center.json');self.complete=RawArray(ROOT/'image_cache/val_complete.json')
    def __len__(self):return len(self.center)
    def __getitem__(self,i):return np.array(self.center[i]),np.array(self.complete[i])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
    folder=ROOT/'runs'/a.stage;began=time.monotonic()
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_model(ROOT,state['config'],ROOT,initialize_official=False).cuda().eval();model.load_state_dict(state['model'],strict=True)
    policy=json.loads((ROOT/'inference_policy.json').read_text());rows=json.loads((ROOT/'data/val_index.json').read_text())
    mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    predicted=[]
    with torch.inference_mode():
        for center,complete in DataLoader(Validation(),batch_size=48,num_workers=6,pin_memory=True):
            arrays={'center':center,'complete':complete};views=[]
            for kind,crop,flip in policy['classification_views']:
                x=arrays[kind].cuda().permute(0,3,1,2).float()/255
                side=max(32,round(512*crop));offset=(512-side)//2;x=x[:,:,offset:offset+side,offset:offset+side]
                x=F.interpolate(x,size=(448,448),mode='bicubic',align_corners=False,antialias=True)
                if flip:x=x.flip(-1)
                with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model((x-mean)/std)
                views.append(logits.float().softmax(-1).cpu().numpy())
            probs=np.maximum(np.stack(views),1e-9)**(1/policy['temperature']);probs/=probs.sum(-1,keepdims=True)
            pred=np.einsum('v,vnc->nc',np.array(policy['view_weights']),probs).argmax(-1);predicted.extend(pred.tolist())
            if len(predicted)%2400==0:print('Validation',a.stage,len(predicted),len(rows),flush=True)
    y=np.array([r[1] for r in rows]);pred=np.array(predicted);assert len(pred)==len(y) and np.isfinite(pred).all()
    correct=np.bincount(y,weights=(y==pred),minlength=500);counts=np.bincount(y,minlength=500)
    report=dict(stage=a.stage,step=state['step'],rows=len(y),accuracy=float((y==pred).mean()),macro_accuracy=float((correct/counts).mean()),seconds=time.monotonic()-began,label_source='held-out original noisy TRAIN labels',test_used=False,policy=policy)
    (folder/'validation.json').write_text(json.dumps(report,indent=2))
    with (folder/'validation_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filename','given_label','prediction']);w.writerows((r[0],f'{int(t):04d}',f'{int(v):04d}') for r,t,v in zip(rows,y,pred))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
