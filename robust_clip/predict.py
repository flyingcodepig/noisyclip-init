"""Single-model fixed four-view inference and submission packaging."""
import argparse,csv,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'engine'))
from model_base import load_model
from robust_v3_support import CacheRows,upload_pixels,view_pixels,attention_context

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--names',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=load_model(a.workspace,state['config'],a.workspace,initialize_official=False)
    model.load_state_dict(state['model'],strict=True);model.cuda().eval()
    names=json.loads(a.names.read_text());names=names['names'] if isinstance(names,dict) else names
    assert len(names)==len(set(names))==len(np.load(a.cache,mmap_mode='r'))
    loader=DataLoader(CacheRows(a.cache,np.arange(len(names))),batch_size=96,num_workers=8,pin_memory=True,prefetch_factor=3)
    predictions=[]
    with torch.inference_mode(),attention_context():
        for raw in loader:
            base=upload_pixels(raw);posterior=None
            for crop,flip,weight in [(1.,False,.3),(1.,True,.3),(.86,False,.2),(.86,True,.2)]:
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(view_pixels(base,state['config']['size'],crop,flip))
                probability=logits.float().softmax(-1).clamp_min(1e-9).pow(1/2.85);probability/=probability.sum(-1,keepdim=True)
                posterior=probability*weight if posterior is None else posterior+probability*weight
            predictions.extend(posterior.argmax(-1).cpu().tolist())
    a.output.mkdir(parents=True,exist_ok=True);target=a.output/'pred_results.csv'
    with target.open('w',newline='',encoding='utf-8') as stream:csv.writer(stream,lineterminator='\n').writerows((n,f'{y:04d}') for n,y in zip(names,predictions))
    assert len(predictions)==len(names) and all(0<=y<500 for y in predictions)
    with zipfile.ZipFile(a.output/'pred_results.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(target,arcname='pred_results.csv')

if __name__=='__main__':main()
