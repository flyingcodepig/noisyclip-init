"""Reuse exact C1 inputs; frozen leave-one-out prototype agreement is TRAIN-only."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
C1=ROOT.parents[1]/'c1_early_lr/code'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def main():
    binding=json.loads((C1/'prepare_manifest.json').read_text())
    for p,h in binding['files'].items():assert sha(C1/p)==h,p
    manifest=ROOT/'prepare_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old['c1_manifest_sha256']==sha(C1/'prepare_manifest.json')
        for p,h in old['files'].items():assert sha(ROOT/p)==h,p
        assert json.loads((ROOT/'stage1.json').read_text())['contrast_queue']==json.loads((ROOT/'stage1.template.json').read_text())['contrast_queue']
        print('Existing C2 inputs verified',flush=True);return
    for d in ['data','cache','image_cache','weights/clip-vit-base-patch32','runs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    for folder in ['data','cache','weights']:
        for p in (C1/folder).rglob('*'):
            if p.is_file():
                target=ROOT/p.relative_to(C1);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in (C1/'engine').glob('*.npy'):shutil.copyfile(p,ROOT/'engine'/p.name)
    for p in (C1/'image_cache').glob('*.json'):
        spec=json.loads(p.read_text());spec['index']=str(ROOT/'data'/Path(spec['index']).name);write(ROOT/'image_cache'/p.name,spec)
    features=np.load(ROOT/'engine/train_teacher.npy');labels=np.load(ROOT/'engine/train_labels.npy')
    assert features.shape==(81177,512) and labels.shape==(81177,)
    sums=np.zeros((500,512),np.float32);np.add.at(sums,labels,features)
    centers=sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-8)
    agreement=np.zeros(len(labels),dtype=bool)
    for start in range(0,len(labels),512):
        f=features[start:start+512];y=labels[start:start+512];scores=f@centers.T
        loo=sums[y]-f;loo/=np.maximum(np.linalg.norm(loo,axis=1,keepdims=True),1e-8)
        scores[np.arange(len(f)),y]=(f*loo).sum(1);agreement[start:start+len(f)]=scores.argmax(-1)==y
    np.save(ROOT/'engine/frozen_loo_agreement.npy',agreement)
    cfg=json.loads((C1/'stage1.json').read_text());template=json.loads((ROOT/'stage1.template.json').read_text())
    for key in ('contrast_queue','class_balanced_beta','ema_decay','ema_drift_weight'):
        if key in template: cfg[key]=template[key]
    write(ROOT/'stage1.json',cfg)
    files={str(p.relative_to(ROOT)):sha(p) for folder in ['data','image_cache','cache','weights'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    for p in list((ROOT/'engine').glob('*.npy'))+[ROOT/'stage1.json']:files[str(p.relative_to(ROOT))]=sha(p)
    write(manifest,dict(c1_manifest_sha256=sha(C1/'prepare_manifest.json'),train=81177,validation=9020,split_identical_to_c1=True,
        frozen_loo_agreement_count=int(agreement.sum()),frozen_loo_agreement_fraction=float(agreement.mean()),test_used=False,files=files))
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(ROOT/'requirements.actual.txt').open('w'),check=True)
    print('C1 split reused; frozen leave-one-out agreement',int(agreement.sum()),len(labels),flush=True)
if __name__=='__main__':main()
