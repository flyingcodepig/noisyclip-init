"""Build the original two lossless uint8 caches directly from semifinal TRAIN."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import io
import json
from pathlib import Path
import shutil
import threading
import zipfile
import numpy as np
from PIL import Image, ImageFile, ImageOps

ROOT = Path(__file__).resolve().parent
ImageFile.LOAD_TRUNCATED_IMAGES = True
LOCAL = threading.local()

def decode(name):
    if not hasattr(LOCAL, 'archive'):
        LOCAL.archive = zipfile.ZipFile(ROOT/'train.zip')
    with Image.open(io.BytesIO(LOCAL.archive.read(name))) as image:
        image = image.convert('RGB')
        complete = np.asarray(ImageOps.pad(image, (512,512), method=Image.Resampling.BICUBIC, color=(123,117,104)))
        w,h = image.size
        scale = 512/min(w,h)
        if (w,h) != (512,512):
            image = image.resize((max(512,round(w*scale)),max(512,round(h*scale))),Image.Resampling.BICUBIC)
        w,h = image.size
        l,t = (w-512)//2,(h-512)//2
        return np.asarray(image.crop((l,t,l+512,t+512))),complete

def main():
    rows = json.loads((ROOT/'data/train_index.json').read_text())
    assert len(rows)==90197 and len(set(row[1] for row in rows))==500
    folder = ROOT/'image_cache'
    folder.mkdir(exist_ok=True)
    targets = [folder/'semi_train_512.npy', folder/'semi_train_complete_512.npy']
    shape = (len(rows),512,512,3)
    if all(p.exists() for p in targets):
        for p in targets:
            a=np.load(p,mmap_mode='r'); assert a.shape==shape and a.dtype==np.uint8
        assert (folder/'TRAIN_CACHE_VERIFIED.json').exists()
        print('Existing semifinal TRAIN caches verified',flush=True)
        return
    required = len(rows)*512*512*3*2
    assert shutil.disk_usage(folder).free > required+10*1024**3, 'Insufficient space for original TRAIN caches'
    temporary = [p.with_suffix('.building.npy') for p in targets]
    arrays = [np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=shape) for p in temporary]
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = deque()
        source = iter(enumerate(rows))
        for _ in range(36):
            i,row=next(source); pending.append((i,pool.submit(decode,row[0])))
        while pending:
            i,future = pending.popleft()
            center,complete=future.result()
            arrays[0][i],arrays[1][i]=center,complete
            item=next(source,None)
            if item is not None:
                j,row=item; pending.append((j,pool.submit(decode,row[0])))
            if (i+1)%2000==0:
                print(json.dumps(dict(stage='train_cache',images=i+1,total=len(rows))),flush=True)
    for a in arrays: a.flush()
    del arrays
    for src,dst in zip(temporary,targets): src.replace(dst)
    metadata=dict(rows=len(rows),shape=shape,source='semifinal TRAIN only',center='short edge 512 then center crop',complete='ImageOps.pad bicubic 512 RGB(123,117,104)',bytes=required)
    (folder/'TRAIN_CACHE_VERIFIED.json').write_text(json.dumps(metadata,indent=2))
    print('Semifinal TRAIN caches completed',flush=True)

if __name__=='__main__': main()
