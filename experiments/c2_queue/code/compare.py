"""Paired held-out comparison against C1 stage1; no competition score claim."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;C1=ROOT.parents[1]/'c1_early_lr/code'
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
a=read(C1/'runs/stage1/validation_predictions.csv');b=read(ROOT/'runs/stage1/validation_predictions.csv')
assert len(a)==len(b)==9020
assert [(r['filename'],r['given_label']) for r in a]==[(r['filename'],r['given_label']) for r in b]
old=np.array([r['given_label']==r['prediction'] for r in a]);new=np.array([r['given_label']==r['prediction'] for r in b])
reports=[json.loads(p.read_text()) for p in [C1/'runs/stage1/validation.json',ROOT/'runs/stage1/validation.json']]
assert reports[0]['step']==reports[1]['step']==4096 and reports[0]['policy']==reports[1]['policy']
out=dict(c1=reports[0]['accuracy'],c2=reports[1]['accuracy'],accuracy_delta_pp=100*float(new.mean()-old.mean()),
    c1_macro=reports[0]['macro_accuracy'],c2_macro=reports[1]['macro_accuracy'],
    corrected=int((~old&new).sum()),regressed=int((old&~new).sum()),both_correct=int((old&new).sum()),both_wrong=int((~old&~new).sum()),
    changed_predictions=sum(x['prediction']!=y['prediction'] for x,y in zip(a,b)),rows=len(a),same_split=True,
    conclusion_boundary='One seed, noisy validation labels; not online score or proof of general superiority')
(ROOT/'runs/comparison_c1.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
