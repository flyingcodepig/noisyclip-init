"""No-gradient readout with frozen training head and explicit class balance."""
import argparse
import csv
import json
import zipfile
from pathlib import Path
import torch
from torch.nn import functional as F

def balance(logits, inner=100):
    q=logits.double()
    n,c=q.shape
    target=torch.log(torch.tensor(n/c,device=q.device,dtype=q.dtype))
    for _ in range(inner):
        q-=torch.logsumexp(q,dim=1,keepdim=True)
        q+=target-torch.logsumexp(q,dim=0,keepdim=True)
    return (q-torch.logsumexp(q,dim=1,keepdim=True)).exp().float()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--feature-power',action='store_true')
    p.add_argument('--logp-weight',type=float,default=1.0)
    p.add_argument('--feature-temperature',type=float,default=0.1)
    a=p.parse_args()
    d=torch.load(a.cache,map_location='cpu',weights_only=False)
    names=d['names']; x=d['feature'].float().cuda(); head=d['head'].float().cuda()
    prob=d['prob'].float().cuda().clamp_min(1e-12).pow(1.1)
    prob/=prob.sum(1,keepdim=True)
    if a.feature_power:
        minimum=x.amin(0,keepdim=True)
        x=F.normalize((x-minimum).clamp_min(1e-8).sqrt(),dim=1)
        head=F.normalize((head-minimum).clamp_min(1e-8).sqrt(),dim=1)
    logits=(F.normalize(x,dim=1)@F.normalize(head,dim=1).T)/a.feature_temperature+a.logp_weight*prob.log()
    labels=balance(logits).argmax(1).cpu().tolist()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'pred_results.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        csv.writer(f,lineterminator='\n').writerows((n,f'{y:04d}') for n,y in zip(names,labels))
    with zipfile.ZipFile(out/'submission.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path,'pred_results.csv')
    print(json.dumps({'images':len(names),'classes_predicted':len(set(labels))}))

if __name__=='__main__': main()
