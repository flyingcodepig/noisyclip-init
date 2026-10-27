import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from fetch_weights import digest, EXPECTED

ROOT=Path(__file__).resolve().parent
ImageFile.LOAD_TRUNCATED_IMAGES=True

class TeacherImages(Dataset):
    def __init__(self, rows, processor):
        self.rows, self.processor, self.archive = rows, processor, None
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if self.archive is None: self.archive=zipfile.ZipFile(ROOT/'train.zip')
        with Image.open(io.BytesIO(self.archive.read(self.rows[i][0]))) as im:
            return self.processor(images=im.convert('RGB'), return_tensors='pt')['pixel_values'][0]

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for folder in ('data','cache','engine','runs'): (ROOT/folder).mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'train.zip') as archive:
        names=sorted(n for n in archive.namelist() if not n.endswith('/'))
        assert len(names)==90197 and len(set(n.split('/')[0] for n in names))==500
        rows=[[n,int(n.split('/')[0])] for n in names]
        (ROOT/'data/train_index.json').write_text(json.dumps(rows))
        labels=np.array([r[1] for r in rows],dtype=np.int64)
        np.save(ROOT/'engine/train_labels.npy',labels)
        groups=defaultdict(list)
        for i,n in enumerate(names):
            info=archive.getinfo(n); groups[(info.file_size,info.CRC)].append(i)
        sets={}
        for key,indices in groups.items():
            if len(indices)>1:
                hashes={hashlib.sha256(archive.read(names[i])).digest() for i in indices}
                assert len(hashes)==1,'CRC collision; exact content verification failed'
            sets[key]=sorted(set(int(labels[i]) for i in indices))
        width=max(map(len,sets.values()))
        candidates=np.full((len(names),width),-1,dtype=np.int64)
        for key,indices in groups.items():
            for i in indices:candidates[i,:len(sets[key])]=sets[key]
        np.save(ROOT/'cache/train_candidate_labels.npy',candidates)
        np.save(ROOT/'cache/placeholder.npy',np.zeros((1,512,512,3),dtype=np.uint8))
        metadata=dict(rows=len(rows),classes=500,max_candidates=width,source='current semifinal TRAIN only, exact-byte SHA256 verified groups',reference_read=False,labels_order_verified=True,candidate_sha256=digest(ROOT/'cache/train_candidate_labels.npy'))
        (ROOT/'cache/train_candidate_labels.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata),flush=True)
    weight=ROOT/'weights/clip-vit-base-patch32/pytorch_model.bin'
    assert digest(weight)==EXPECTED,'Official CLIP checksum mismatch'
    folder=weight.parent
    processor=CLIPImageProcessor.from_pretrained(folder,local_files_only=True)
    model=CLIPVisionModelWithProjection.from_pretrained(folder,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    features_path=ROOT/'engine/train_teacher.npy'
    if features_path.exists():
        features=np.load(features_path)
        assert features.shape==(len(rows),512)
    else:
        loader=DataLoader(TeacherImages(rows,processor),batch_size=128,num_workers=6,pin_memory=True)
        parts=[]
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for i,pixels in enumerate(loader):
                features=F.normalize(model(pixels.cuda()).image_embeds,dim=-1)
                parts.append(features.half().cpu().float().numpy())
                if i%25==0:print('Official TRAIN features',min((i+1)*128,len(rows)),len(rows),flush=True)
        features=np.concatenate(parts).astype(np.float32)
        features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
        np.save(features_path,features)
    centers=np.zeros((500,512),np.float32)
    np.add.at(centers,labels,features)
    centers/=np.bincount(labels,minlength=500)[:,None]
    centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    np.save(ROOT/'engine/initial_centers.npy',centers)
    manifest={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'train.zip',features_path,ROOT/'engine/train_labels.npy',ROOT/'engine/initial_centers.npy',ROOT/'cache/train_candidate_labels.npy',weight]}
    (ROOT/'semifinal_input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Semifinal TRAIN inputs ready',flush=True)

if __name__=='__main__': main()
