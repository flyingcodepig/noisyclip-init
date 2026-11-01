import json, hashlib, os
from pathlib import Path
from collections import defaultdict, Counter
ROOT=Path('/root/aic_a100/eat_semifinal')
source=Path('/root/aic_a100/replica_a/data/train')
groups=defaultdict(list)
for p in sorted(source.glob('*/*.jpg')):
    groups[hashlib.sha256(p.read_bytes()).hexdigest()].append((p,int(p.parent.name)))
assert sum(map(len,groups.values()))==90197
buckets=defaultdict(list)
for digest,rows in groups.items():
    buckets[min(y for _,y in rows)].append(digest)
held=set()
for label,digests in buckets.items():
    digests.sort(key=lambda d:hashlib.sha256(('eat-semifinal-41291:'+d).encode()).digest())
    target=round(sum(len(groups[d]) for d in digests)*.1)
    count=0
    for d in digests:
        if count>=target: break
        held.add(d); count+=len(groups[d])
train=ROOT/'split/train'; val=ROOT/'split/validation'
val.mkdir(parents=True,exist_ok=True)
for c in range(500): (train/f'{c:04d}').mkdir(parents=True,exist_ok=True)
manifest=[]; labels={}; train_count=0
for d,rows in groups.items():
    split='validation' if d in held else 'train'
    for p,y in rows:
        target=(val/f'{y:04d}_{p.name}') if split=='validation' else (train/f'{y:04d}'/p.name)
        if not target.exists(): target.symlink_to(p)
        manifest.append({'source':str(p),'sha256':d,'label':y,'split':split})
        if split=='validation': labels[target.name]=f'{y:04d}'
        else: train_count+=1
assert set(x['sha256'] for x in manifest if x['split']=='train').isdisjoint(held)
assert all(any((train/f'{c:04d}').iterdir()) for c in range(500))
(ROOT/'split/manifest.json').write_text(json.dumps(manifest))
(ROOT/'split/validation_labels.json').write_text(json.dumps(labels))
cfg=json.loads(Path('/root/aic_a100/replica_a/config.json').read_text())
ratio=train_count/90197
for k in ['main_steps','tail_steps','warmup_steps','soft_start','soft_ramp','soft_max_gap']:
    cfg[k]=max(1,round(cfg[k]*ratio))
cfg.update(expected_train=train_count,expected_test=len(labels),workers=8)
(ROOT/'config.baseline.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'train':train_count,'validation':len(labels),'classes':500,'config':cfg}),flush=True)
