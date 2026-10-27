import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageFile, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.data.image_cache import _load_one, _init_worker


def complete(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.pad(im.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC,color=(123,117,104)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('train','test'), required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--cache-dir', type=Path, default=ROOT/'image_cache')
    a = p.parse_args()
    official = Path(json.loads((ROOT/'data/a100/source_manifest.json').read_text())['splits'][a.split]['root'])
    if a.split == 'train':
        index = json.loads((ROOT/'data/a100/train_index.json').read_text())
        paths = [official/r[0] for r in index]
    else:
        names = json.loads((ROOT/'data/a100/test_names.json').read_text())['names']
        paths = [official/name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} official images; examples: {missing[:3]}. Place official images under data/official')
    _init_worker(512)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    out = a.cache_dir
    out.mkdir(exist_ok=True)
    for kind in ('center','complete'):
        target = out/f"semi_{a.split}{'_complete' if kind=='complete' else ''}_512.npy"
        temporary = target.with_suffix('.building.npy')
        if temporary.exists():
            raise FileExistsError(temporary)
        if target.exists():
            cached = np.load(target, mmap_mode='r')
            assert cached.shape == (len(paths),512,512,3) and cached.dtype == np.uint8
            print('Existing cache:', target, flush=True)
            continue
        array = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(len(paths),512,512,3))
        def decode(path):
            if kind == 'complete':
                return complete(path)
            pixels, ok = _load_one(str(path))
            if not ok:
                raise ValueError(f'Cannot decode {path}')
            return pixels
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            # Bound completed decoded images; Executor.map in Python 3.11 eagerly
            # submits all 90k images and can otherwise retain tens of GB.
            for start in range(0, len(paths), a.workers * 4):
                for offset, pixels in enumerate(pool.map(decode, paths[start:start + a.workers * 4])):
                    i = start + offset
                    array[i] = pixels
                    if (i+1)%10000 == 0:
                        print(a.split,kind,i+1,flush=True)
        array.flush()
        del array
        temporary.rename(target)


if __name__ == '__main__':
    main()
