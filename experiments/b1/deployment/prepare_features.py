"""New-dataset frozen official features with explicit, independently checked provenance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(ROOT))
from prepare_training import TrainingImages
from fetch_weights import EXPECTED, digest

def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT / 'weights/clip-vit-base-patch32'
    assert digest(folder / 'pytorch_model.bin') == EXPECTED
    out = ROOT / 'cache/a100'
    out.mkdir(parents=True, exist_ok=True)
    recipe = out / 'feature_provenance.json'
    if recipe.exists():
        record = json.loads(recipe.read_text())
        for path, expected in record['hashes'].items():
            assert digest(ROOT / path) == expected, path
        print('Existing actual-data features verified', flush=True)
        return
    rows = json.loads((ROOT / 'data/a100/train_index.json').read_text())
    labels = np.asarray([r[1] for r in rows], dtype=np.int64)
    assert set(labels) == set(range(500))
    processor = CLIPImageProcessor.from_pretrained(folder, local_files_only=True)
    model = CLIPVisionModelWithProjection.from_pretrained(folder, local_files_only=True, attn_implementation='eager').cuda().eval()
    ds = TrainingImages([ROOT / 'data/official/train' / r[0] for r in rows], processor)
    loader = DataLoader(ds, batch_size=512, num_workers=8, pin_memory=True)
    arrays = []
    began = time.monotonic()
    with torch.inference_mode():
        for no, pixels in enumerate(loader):
            embedding = model(pixels.cuda()).image_embeds
            values = F.normalize(embedding, dim=-1).half().cpu().float().numpy()
            if no == 0:
                repeat = F.normalize(model(pixels.cuda()).image_embeds, dim=-1).half().cpu().float().numpy()
                assert np.array_equal(values, repeat), 'Feature reproducibility check failed'
            arrays.append(values)
            if no % 10 == 0:
                print(json.dumps({'stage': 'official_features', 'rows': min((no + 1) * 512, len(rows)), 'total': len(rows), 'seconds': time.monotonic() - began}), flush=True)
    features = np.concatenate(arrays).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    centers = np.zeros((500, 512), np.float32)
    np.add.at(centers, labels, features)
    centers /= np.bincount(labels, minlength=500)[:, None]
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    assert np.isfinite(features).all() and np.isfinite(centers).all()
    for name, value in [('train_teacher', features), ('train_labels', labels), ('initial_centers', centers)]:
        temp = out / f'{name}.building.npy'
        np.save(temp, value)
        temp.replace(out / f'{name}.npy')
    paths = ['data/a100/train_index.json', 'data/a100/train_sha256.json', 'data/a100/source_manifest.json', 'weights/clip-vit-base-patch32/pytorch_model.bin']
    paths += [f'cache/a100/{name}.npy' for name in ['train_teacher', 'train_labels', 'initial_centers']]
    record = {'source': 'Actual Matpool TRAIN, frozen official CLIP only', 'old_148695_recipe_reused': False, 'batch': 512, 'attention': 'eager', 'precision': 'fp32 then fp16 roundtrip per original recipe', 'first_batch_exact_repeat': True, 'rows': len(rows), 'classes': 500, 'torch': torch.__version__, 'hashes': {p: digest(ROOT / p) for p in paths}, 'seconds': time.monotonic() - began}
    recipe.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
