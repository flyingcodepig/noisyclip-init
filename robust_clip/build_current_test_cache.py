"""Decode current test.zip directly into original, bounded-memory image caches."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib, io, json, shutil, threading, zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageFile,ImageOps
ROOT=Path(__file__).resolve().parent
LOCAL=threading.local()
ImageFile.LOAD_TRUNCATED_IMAGES=True

def decode(name):
    if not hasattr(LOCAL,'archive'): LOCAL.archive=zipfile.ZipFile(ROOT/'test.zip')
    with Image.open(io.BytesIO(LOCAL.archive.read(name))) as image:
        image=image.convert('RGB')
        complete=np.asarray(ImageOps.pad(image,(512,512),method=Image.Resampling.BICUBIC,color=(123,117,104)))
        w,h=image.size; scale=512/min(w,h)
        if (w,h)!=(512,512): image=image.resize((max(512,round(w*scale)),max(512,round(h*scale))),Image.Resampling.BICUBIC)
        w,h=image.size; l,t=(w-512)//2,(h-512)//2
        return np.asarray(image.crop((l,t,l+512,t+512))),complete

def main():
    assert (ROOT/'TEST_UPLOAD_VERIFIED').exists()
    with zipfile.ZipFile(ROOT/'test.zip') as archive:
        names=sorted(n for n in archive.namelist() if not n.endswith('/'))
        assert len(names)==len(set(names))==49857
        assert all(Path(n).suffix.lower() in ('.jpg','.jpeg','.png') for n in names)
    target_names=ROOT/'data/test_names.json'
    if target_names.exists() and not (ROOT/'data/test_names_before_current_test.json').exists():
        shutil.copy2(target_names,ROOT/'data/test_names_before_current_test.json')
    target_names.write_text(json.dumps(dict(names=names,source='user current test.zip')))
    folder=ROOT/'image_cache'; shape=(len(names),512,512,3)
    targets=[folder/'semi_test_512.npy',folder/'semi_test_complete_512.npy']
    if all(p.exists() for p in targets):
        assert (folder/'CURRENT_TEST_CACHE_VERIFIED.json').exists()
        for p in targets: assert np.load(p,mmap_mode='r').shape==shape
        return
    assert not any(p.exists() for p in targets),'Partial completed caches require inspection'
    assert shutil.disk_usage(folder).free>len(names)*512*512*3*2+10*1024**3
    temporary=[p.with_suffix('.building.npy') for p in targets]
    assert not any(p.exists() for p in temporary),'Interrupted build requires inspection'
    arrays=[np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=shape) for p in temporary]
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending=deque(); source=iter(enumerate(names))
        for _ in range(36):
            i,name=next(source);pending.append((i,pool.submit(decode,name)))
        while pending:
            i,future=pending.popleft();center,complete=future.result()
            arrays[0][i],arrays[1][i]=center,complete
            item=next(source,None)
            if item:
                j,name=item;pending.append((j,pool.submit(decode,name)))
            if (i+1)%2000==0:print(json.dumps(dict(stage='test_cache',images=i+1,total=len(names))),flush=True)
    for array in arrays:array.flush()
    del arrays
    for source,target in zip(temporary,targets):source.replace(target)
    (folder/'CURRENT_TEST_CACHE_VERIFIED.json').write_text(json.dumps(dict(rows=len(names),shape=shape,names_sha256=hashlib.sha256(target_names.read_bytes()).hexdigest()),indent=2))
    print('Current test caches ready',flush=True)

if __name__=='__main__':main()
