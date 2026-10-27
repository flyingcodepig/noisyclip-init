"""Lossless, on-demand equivalent of the original two 512px image caches."""
import io
import json
import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile, ImageOps
try:
    from data_stream import StreamDataset
except ModuleNotFoundError:
    from engine.data_stream import StreamDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent

class ZipArray:
    def __init__(self, names, kind='center'):
        self.names, self.kind = names, kind
        self.shape = (len(names), 512, 512, 3)
        self.archive = None
        self.pid = None

    def read(self, index):
        if self.pid != os.getpid():
            self.archive = zipfile.ZipFile(ROOT / 'train.zip')
            self.pid = os.getpid()
        return self.archive.read(self.names[index])

    def __getitem__(self, index):
        with Image.open(io.BytesIO(self.read(index))) as image:
            image = image.convert('RGB')
            if self.kind == 'complete':
                image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC, color=(123,117,104))
            else:
                w, h = image.size
                s = 512 / min(w, h)
                if (w, h) != (512, 512):
                    image = image.resize((max(512, round(w*s)), max(512, round(h*s))), Image.Resampling.BICUBIC)
                w, h = image.size
                l, t = (w-512)//2, (h-512)//2
                image = image.crop((l,t,l+512,t+512))
            return np.asarray(image, dtype=np.uint8)

class RawStreamDataset(StreamDataset):
    def __init__(self, cfg, stream_states=None):
        rows = json.loads((ROOT/'data/train_index.json').read_text())
        labels = [row[1] for row in rows]
        super().__init__(ROOT/'cache/placeholder.npy', np.arange(len(rows)), labels,
            size=cfg['size'], train=True, scale=(cfg['rrc_min'],1),
            wideshot_p=cfg['wideshot_p'], wideshot_scale=cfg['wideshot_scale'],
            wideshot_feather=cfg['feather'], jitter=cfg['jitter'],
            randaug_n=cfg['randaug_n'], randaug_m=cfg['randaug_m'],
            seed_mode='worker', base_seed=cfg['seed'], workers=cfg['workers'],
            stream_states=stream_states, alternate_probability=cfg['alternate_complete_probability'])
        names = [row[0] for row in rows]
        self.arr = ZipArray(names)
        self.alternate = ZipArray(names, 'complete')

