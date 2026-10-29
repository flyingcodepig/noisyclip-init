"""Prepare isolated training workspaces; never copy full image caches or old trained weights."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'optimization_v1'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def patch_trainer():
    source=(ROOT/'engine/train_stage1.py').read_text()
    replacements=[
      ("    groups = {'backbone': [], 'mechanism': [], 'head': []}","    from optimization_support import configure_optimizer_scope,set_training_phase,adjust_unfreeze_lr,load_training_indices,evaluate_holdout\n    configure_optimizer_scope(model,cfg)\n    groups = {'backbone': [], 'mechanism': [], 'head': []}"),
      ("    labels = np.load(HERE / 'train_labels.npy')","    labels = np.load(HERE / 'train_labels.npy')\n    global_indices=load_training_indices(ROOT,cfg,labels)"),
      ("ds = dataset_class(training_cache, np.arange(len(labels)), labels.tolist()","ds = dataset_class(training_cache, global_indices, labels.tolist()"),
      ("            perm = torch.randperm(len(y), device=y.device)","            set_training_phase(model,cfg,step,out)\n            perm = torch.randperm(len(y), device=y.device)"),
      ("                group['lr'] = group['initial_lr'] * fraction","                group['lr'] = group['initial_lr'] * fraction\n                adjust_unfreeze_lr(group,cfg,step)"),
      ("            if step % cfg['save_every'] == 0 or step == args.until:\n                save()","            if step % cfg['save_every'] == 0 or step == args.until:\n                save()\n                evaluate_holdout(model,cfg,ROOT,out,step,anchors)"),
      ("    if rank == 0:\n        print(json.dumps(dict(stage='completed'","    if rank == 0:\n        evaluate_holdout(model,cfg,ROOT,out,step,anchors)\n        print(json.dumps(dict(stage='completed'")]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    compile(source,'train_optimization.py','exec')
    return source

def prepare(name,selected,validation,source_labels,source_features,configs):
    target=BASE/name;target.mkdir(parents=True,exist_ok=True)
    if (target/'PREPARED.json').exists():
        assert np.array_equal(np.load(target/'data/training_global_indices.npy'),selected)
        assert np.array_equal(np.load(target/'data/validation_global_indices.npy'),validation)
        for n,cfg in configs.items():assert json.loads((target/n).read_text())==cfg
        return target
    for folder in ['data','cache','engine','runs','logs']:(target/folder).mkdir(exist_ok=True)
    shutil.copytree(ROOT/'src',target/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for path in (ROOT/'engine').glob('*.py'):shutil.copy2(path,target/'engine'/path.name)
    shutil.copy2(ROOT/'optimization_support.py',target/'optimization_support.py')
    (target/'engine/train_optimization.py').write_text(patch_trainer())
    for folder in ['weights','image_cache']:
        link=target/folder
        if not link.exists():link.symlink_to(ROOT/folder,target_is_directory=True)
    labels=source_labels[selected];features=source_features[selected].copy()
    centers=np.zeros((500,512),np.float32);np.add.at(centers,labels,features)
    counts=np.bincount(labels,minlength=500)
    if name!='smoke':assert (counts>0).all()
    centers/=np.maximum(counts[:,None],1);centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    if name=='smoke':
        # Missing fixture classes get centers fitted on TRAIN subset, never on holdout.
        trained=np.load(BASE/'B0/engine/initial_centers.npy');centers[counts==0]=trained[counts==0]
    np.save(target/'data/training_global_indices.npy',selected)
    np.save(target/'data/validation_global_indices.npy',validation)
    np.save(target/'data/source_labels.npy',source_labels)
    np.save(target/'engine/train_labels.npy',labels)
    np.save(target/'engine/train_teacher.npy',features)
    np.save(target/'engine/initial_centers.npy',centers)
    np.save(target/'cache/train_candidate_labels.npy',labels[:,None])
    assert not np.intersect1d(selected,validation).size
    metadata=dict(rows=len(selected),classes=500,reference_read=False,labels_order_verified=True,training_only=True)
    (target/'cache/train_candidate_labels.json').write_text(json.dumps(metadata,indent=2))
    sha=digest(target/'cache/train_candidate_labels.npy')
    for n,cfg in configs.items():
        cfg['candidate_labels_sha256']=sha
        (target/n).write_text(json.dumps(cfg,indent=2))
    report=dict(name=name,train_rows=len(selected),validation_rows=len(validation),overlap_rows=0,
        centers_from_train_only=True,teacher='reused label-independent frozen official features, indexed by train rows',
        old_finetuned_weights_used=False,train_indices_sha256=digest(target/'data/training_global_indices.npy'),
        validation_indices_sha256=digest(target/'data/validation_global_indices.npy'))
    (target/'PREPARED.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return target

def main():
    BASE.mkdir(exist_ok=True)
    assert digest(ROOT/'weights/clip-vit-base-patch32/pytorch_model.bin')=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    train=np.load(ROOT/'validation_seen/remaining_indices.npy');validation=np.load(ROOT/'validation_seen/indices.npy')
    assert len(train)==81165 and len(validation)==9032
    labels=np.load(ROOT/'engine/train_labels.npy');features=np.load(ROOT/'engine/train_teacher.npy',mmap_mode='r')
    assert len(np.union1d(train,validation))==len(labels)==90197
    assert np.load(ROOT/'cache/train_candidate_labels.npy').shape==(90197,1)
    for name in ['B0','B1']:
        configs={}
        for stage,source,steps in [(1,'semifinal_stage1.json',3686),(2,'semifinal_stage2.json',2765)]:
            cfg=json.loads((ROOT/source).read_text());cfg.update(total_steps=steps,save_every=256,training_indices='data/training_global_indices.npy',validation_indices='data/validation_global_indices.npy')
            if name=='B1':cfg.update(gradual_unfreeze=True,unfreeze_last_blocks=4,head_warmup_steps=422 if stage==1 else 0,unfreeze_warmup_steps=128,lr_backbone=8e-6 if stage==1 else 3.5e-6)
            cfg['candidate_labels_sha256']=digest_candidate(labels[train])
            configs[f'stage{stage}.json']=cfg
        prepare(name,train,validation,labels,features,configs)
    cfg=json.loads((BASE/'B1/stage1.json').read_text());cfg.update(head_warmup_steps=1,workers=2,save_every=1,total_steps=4)
    cfg['candidate_labels_sha256']=digest_candidate(labels[train[:512]])
    prepare('smoke',train[:512],validation[:24],labels,features,{'smoke.json':cfg})

def digest_candidate(labels):
    import io
    stream=io.BytesIO();np.save(stream,labels[:,None]);return hashlib.sha256(stream.getvalue()).hexdigest()

if __name__=='__main__':main()
