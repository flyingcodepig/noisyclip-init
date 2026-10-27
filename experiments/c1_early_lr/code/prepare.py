"""Fixed stratified 10% split; TRAIN-only centers, frozen per-image features reused."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
B1=ROOT.parents[1]/'b1/code'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p.write_text(json.dumps(obj,indent=2)+'\n')

def main():
    manifest=ROOT/'prepare_manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        for p,h in m['files'].items():assert sha(ROOT/p)==h,p
        print('Existing split and inputs verified',flush=True);return
    for d in ['data','cache','image_cache','weights/clip-vit-base-patch32','runs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    index=B1/'data/a100/train_index.json';hashfile=B1/'data/a100/train_sha256.json'
    assert sha(index)=='2e729256ea5a2393a852b2bf50aa2bd060ff9e4f8ede20308e971d5d9a90250f'
    assert sha(hashfile)=='176011b88499e7b20c27152d9f70723934ebd7789cf21fb0ce235014c8ee7283'
    rows=json.loads(index.read_text());hashes=dict(json.loads(hashfile.read_text()))
    assert len(rows)==90197 and len(set(hashes.values()))==len(rows), 'Non-unique groups require group-aware allocation'
    labels=np.array([r[1] for r in rows]);counts=np.bincount(labels,minlength=500)
    assert len(counts)==500 and counts.min()>1
    quotas=np.floor(counts*.1).astype(int);target=round(len(rows)*.1)
    order=np.argsort(-(counts*.1-quotas),kind='stable');quotas[order[:target-int(quotas.sum())]]+=1
    rng=np.random.default_rng(20261028);valid=[]
    for c in range(500):valid.extend(rng.permutation(np.flatnonzero(labels==c))[:quotas[c]].tolist())
    valid=np.array(sorted(valid));train=np.setdiff1d(np.arange(len(rows)),valid)
    assert len(valid)==9020 and len(train)==81177
    assert not ({hashes[rows[i][0]] for i in train}&{hashes[rows[i][0]] for i in valid})
    source=json.loads((B1/'data/a100/source_manifest.json').read_text())['splits']['train']['root']
    for j,(name,_) in enumerate(rows):
        assert sha(Path(source)/name)==hashes[name],name
        if j%10000==0:print('Source integrity',j,len(rows),flush=True)
    for split,ids in [('train',train),('val',valid)]:
        write(ROOT/f'data/{split}_index.json',[rows[int(i)] for i in ids]);np.save(ROOT/f'data/{split}_ids.npy',ids)
        for kind in ['center','complete']:
            name=kind if split=='train' else 'val_'+kind
            write(ROOT/f'image_cache/{name}.json',dict(root=source,index=str(ROOT/f'data/{split}_index.json'),kind=kind))
    feature=B1/'cache/a100/train_teacher.npy'
    assert sha(feature)=='cccae8eaee13bfb50d631420227c36dc83001bb93e618a835b5a6e564308ea9c'
    assert sha(B1/'weights/clip-vit-base-patch32/pytorch_model.bin')=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    features=np.load(feature)[train];y=labels[train]
    assert features.shape==(81177,512) and np.isfinite(features).all()
    centers=np.zeros((500,512),np.float32);np.add.at(centers,y,features)
    centers/=np.bincount(y,minlength=500)[:,None];centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    for name,a in [('train_teacher',features),('train_labels',y),('initial_centers',centers)]:np.save(ROOT/f'engine/{name}.npy',a)
    np.save(ROOT/'cache/train_candidate_labels.npy',y[:,None])
    write(ROOT/'cache/train_candidate_labels.json',dict(reference_read=False,labels_order_verified=True,rows=len(y),max_candidates=1))
    for p in (B1/'weights/clip-vit-base-patch32').iterdir():
        if p.is_file() and p.suffix in ['.json','.bin']:shutil.copyfile(p,ROOT/'weights/clip-vit-base-patch32'/p.name)
    for stage in [1,2]:
        c=json.loads((ROOT/f'stage{stage}.template.json').read_text());c['candidate_labels_sha256']=sha(ROOT/'cache/train_candidate_labels.npy')
        write(ROOT/f'stage{stage}.json',c)
    files={str(p.relative_to(ROOT)):sha(p) for folder in ['data','image_cache','cache','weights'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    for p in list((ROOT/'engine').glob('*.npy'))+list(ROOT.glob('stage?.json')):files[str(p.relative_to(ROOT))]=sha(p)
    write(manifest,dict(seed=20261028,total=len(rows),train=len(train),validation=len(valid),validation_fraction=len(valid)/len(rows),exact_duplicate_isolation=True,near_duplicate_detection=False,per_class_validation=quotas.tolist(),centers_train_only=True,feature_source='frozen official CLIP per-image features from B1 preparation; no B1 trained weights',files=files))
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(ROOT/'requirements.actual.txt').open('w'),check=True)
    print('Prepared 81177 training / 9020 validation; 500 classes',flush=True)

if __name__=='__main__':main()
