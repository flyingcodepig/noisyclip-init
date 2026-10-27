"""Engineering-only resume check on an isolated subset of semifinal TRAIN.

All generated checkpoints stay in the isolated smoke workspace and are never
used to initialize the real competition run.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def main():
    target=ROOT/'engineering_smoke_v3'
    assert not target.exists(), 'Use the saved smoke results instead of overwriting them'
    target.mkdir()
    for name in ['src','engine']:
        shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('*.npy','__pycache__'))
    shutil.copy(ROOT/'semi_data.py',target/'semi_data.py')
    (target/'weights').symlink_to(ROOT/'weights',target_is_directory=True)
    shutil.copy(ROOT/'smoke_train_subset.zip',target/'train.zip')
    for name in ['data','cache']:(target/name).mkdir()
    with zipfile.ZipFile(target/'train.zip') as archive:
        names=sorted(n for n in archive.namelist() if not n.endswith('/'))
        rows=[[n,int(n.split('/')[0])] for n in names]*4
    assert len(rows)==512
    (target/'data/train_index.json').write_text(json.dumps(rows))
    labels=np.array([r[1] for r in rows],dtype=np.int64)
    np.save(target/'engine/train_labels.npy',labels)
    # Fixed normalized synthetic targets exercise the loss without reusing any
    # smoke outputs in real training. The image inputs are official TRAIN only.
    rng=np.random.default_rng(123)
    teacher=rng.standard_normal((len(rows),512)).astype(np.float32)
    teacher/=np.linalg.norm(teacher,axis=1,keepdims=True)
    centers=rng.standard_normal((500,512)).astype(np.float32)
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    np.save(target/'engine/train_teacher.npy',teacher)
    np.save(target/'engine/initial_centers.npy',centers)
    np.save(target/'cache/placeholder.npy',np.zeros((1,512,512,3),dtype=np.uint8))
    np.save(target/'cache/train_candidate_labels.npy',labels[:,None])
    from fetch_weights import digest
    (target/'cache/train_candidate_labels.json').write_text(json.dumps(dict(reference_read=False,labels_order_verified=True,purpose='isolated engineering smoke test')))
    batch=json.loads((ROOT/'hardware_profile.json').read_text())['microbatch']
    cfg=json.loads((ROOT/'train_config_stage1.json').read_text())
    cfg.update(batch=batch,gradient_accumulation=192//batch,workers=2,total_steps=3,save_every=1,num_classes=500,data_backend='zip',candidate_labels_sha256=digest(target/'cache/train_candidate_labels.npy'))
    config=target/'test_config.json'; config.write_text(json.dumps(cfg))
    command=[sys.executable,str(target/'engine/train_stage1.py'),'--workspace',str(target),'--config',str(config)]
    subprocess.run(command+['--run','uninterrupted','--until','3'],check=True)
    subprocess.run(command+['--run','resumed','--until','1'],check=True)
    subprocess.run(command+['--run','resumed','--until','3','--resume'],check=True)
    a=torch.load(target/'runs/uninterrupted/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(target/'runs/resumed/last.pt',map_location='cpu',weights_only=False)
    different=[k for k in a['model'] if not torch.equal(a['model'][k],b['model'][k])]
    assert not different, different
    report=dict(passed=True,step=3,model_tensors=len(a['model']),bitwise_identical=True,microbatch=batch,effective_batch=192,checked='mid-epoch resume and epoch transition',scope='engineering check only; checkpoints never used for official training')
    (ROOT/'resume_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
