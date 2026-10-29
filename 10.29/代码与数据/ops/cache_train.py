"""Rebuild the exact RGB512 training canvases in the recorded image order."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.runtime import file_hash, save, PIXEL_SHA, FEATURE_SHA, MANIFEST_SHA

def pixel_canvas(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        scale = 512 / min(width, height)
        if (width, height) != (512, 512):
            image = image.resize((max(512, int(round(width * scale))),
                                  max(512, int(round(height * scale)))), Image.Resampling.BICUBIC)
        width, height = image.size
        left, top = (width - 512) // 2, (height - 512) // 2
        return np.array(image.crop((left, top, left + 512, top + 512)), copy=True)

def prepare(train_dir, cache, workers=8, check_files_only=False):
    manifest = ROOT / 'inputs/train_manifest.jsonl'
    if file_hash(manifest) != MANIFEST_SHA:
        raise ValueError('TRAIN manifest checksum changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    paths = [train_dir / row['relative_path'] for row in rows]
    if len(rows) != 90197 or set(row['label'] for row in rows) != set(range(500)):
        raise ValueError('Unexpected TRAIN labels or image count')
    existing = sorted(p.relative_to(train_dir).as_posix() for p in train_dir.glob('*/*.jpg'))
    if existing != sorted(row['relative_path'] for row in rows):
        raise ValueError('TRAIN files must exactly match the manifest')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, digest in zip(rows, pool.map(file_hash, paths)):
            if row['sha256'] != digest:
                raise ValueError('Original TRAIN file changed: ' + row['relative_path'])
    print('Verified all 90197 original TRAIN files', flush=True)
    if check_files_only:
        return
    cache.mkdir(parents=True, exist_ok=True)
    feature = ROOT / 'inputs/official_clip224_features.npy'
    if file_hash(feature) != FEATURE_SHA:
        raise ValueError('Official untrained CLIP features changed')
    target_features = cache / feature.name
    if not target_features.exists():
        shutil.copy2(feature, target_features)
    elif file_hash(target_features) != FEATURE_SHA:
        raise ValueError('An existing feature cache differs')
    pixel = cache / 'pixel512_train.npy'
    if not pixel.exists():
        if shutil.disk_usage(cache).free < 70933807232 + 2**30:
            raise OSError('At least 68 GiB free space is required for the RGB512 cache')
        temporary = cache / 'pixel512_train.writing.npy'
        if temporary.exists():
            raise FileExistsError('An incomplete cache exists; use another empty cache directory')
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(90197, 512, 512, 3))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(paths), 128):
                for offset, value in enumerate(pool.map(pixel_canvas, paths[start:start+128])):
                    array[start + offset] = value
                if start % 8192 == 0:
                    print(f'RGB512 cache: {start}/{len(paths)}', flush=True)
        array.flush()
        del array
        if file_hash(temporary) != PIXEL_SHA:
            raise ValueError('Rebuilt pixel cache differs from the recorded original-image decoding')
        temporary.replace(pixel)
    elif file_hash(pixel) != PIXEL_SHA:
        raise ValueError('The existing pixel cache changed')
    save(cache / 'cache_receipt.json', dict(status='complete', images=90197,
        manifest_sha256=MANIFEST_SHA, pixels_sha256=PIXEL_SHA, features_sha256=FEATURE_SHA))
    print('TRAIN cache verified', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=ROOT / 'data/semifinal/train')
    parser.add_argument('--cache', type=Path, default=ROOT / 'var/cache/train')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-files-only', action='store_true')
    args = parser.parse_args()
    prepare(args.train_dir, args.cache, args.workers, args.check_files_only)
