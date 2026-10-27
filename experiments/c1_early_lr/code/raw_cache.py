"""Decode the archived 512px cache recipe lazily; no random draws here."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RawArray:
    def __init__(self, path):
        self.spec = json.loads(Path(path).read_text())
        self.rows = json.loads(Path(self.spec['index']).read_text())
        self.shape = (len(self.rows),512,512,3)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        if not np.isscalar(i):
            return np.stack([self[int(j)] for j in i])
        with Image.open(Path(self.spec['root'])/self.rows[int(i)][0]) as im:
            im = im.convert('RGB')
            if self.spec['kind']=='complete':
                return np.asarray(ImageOps.pad(im,(512,512),method=Image.Resampling.BICUBIC,color=(123,117,104)))
            w,h=im.size;s=512/min(w,h)
            if (w,h)!=(512,512):
                im=im.resize((max(512,round(w*s)),max(512,round(h*s))),Image.Resampling.BICUBIC)
            w,h=im.size;l,t=(w-512)//2,(h-512)//2
            return np.asarray(im.crop((l,t,l+512,t+512)))

def load_cache(path):
    return RawArray(path) if Path(path).suffix=='.json' else np.load(path,mmap_mode='r')
