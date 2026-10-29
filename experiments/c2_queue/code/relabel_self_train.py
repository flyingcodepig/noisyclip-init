"""Generate model-derived soft labels (self-training) from a trained checkpoint.

Runs a trained single model (default: the stage1_v2 SOTA checkpoint) over the
TRAIN split with a single center view, then builds a soft label for every sample
as a convex combination of the one-hot given label and the model's temperature-
scaled prediction. This is automatic, TRAIN-only and code-reproducible (no
manual cleaning, no TEST data).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from model_base import load_model
from raw_cache import RawArray

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')[None, :, None, None]
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')[None, :, None, None]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TrainCenter(Dataset):
    def __init__(self):
        self.center = RawArray(ROOT / 'image_cache' / 'center.json')

    def __len__(self):
        return len(self.center)

    def __getitem__(self, i):
        return np.array(self.center[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs' / 'stage1_v2' / 'model.pt')
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--alpha', type=float, default=0.3, help='mixing weight toward the model prediction')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--weight', type=float, default=0.5, help='soft_label_weight written into the generated config')
    parser.add_argument('--proto-weight', type=float, default=0.1, help='prototype_contrast_weight written into the config')
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    labels = np.load(ENGINE / 'train_labels.npy')
    num_classes = cfg.get('num_classes', 500)
    assert len(labels) == 81177 and labels.shape == (81177,)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(ROOT, cfg, ROOT, initialize_official=False).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    del state

    began = time.monotonic()
    probs = np.empty((len(labels), num_classes), dtype=np.float32)
    loader = DataLoader(TrainCenter(), batch_size=args.batch, num_workers=6, pin_memory=True)
    assert len(loader.dataset) == len(labels), f'train cache rows {len(loader.dataset)} != labels {len(labels)}'
    with torch.inference_mode():
        for number, batch in enumerate(loader):
            x = batch.cuda().permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(448, 448), mode='bicubic', align_corners=False, antialias=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model((x - MEAN) / STD)
            probs[number * args.batch:number * args.batch + len(batch)] = logits.float().softmax(-1).cpu().numpy()
            if number % 50 == 0:
                print(json.dumps(dict(stage='train_inference', images=number * args.batch, total=len(labels))), flush=True)
    del model
    torch.cuda.empty_cache()

    soft = np.zeros((len(labels), num_classes), dtype=np.float32)
    soft[np.arange(len(labels)), labels] = 1.0
    soft = (1.0 - args.alpha) * soft + args.alpha * probs

    out = ROOT / 'cache' / 'model_soft_labels.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, soft.astype(np.float32))

    manifest = dict(
        rows=int(len(labels)), classes=int(num_classes),
        alpha=args.alpha, temperature=args.temperature,
        soft_label_weight=args.weight, prototype_contrast_weight=args.proto_weight,
        checkpoint=str(checkpoint), test_used=False,
        labels_sha256=sha(ENGINE / 'train_labels.npy'),
        soft_labels_sha256=sha(out),
        seconds=round(time.monotonic() - began, 1),
    )
    (out.with_suffix('.json')).write_text(json.dumps(manifest, indent=2) + '\n')

    # Emit a ready-to-run config for the aggressive self-training run.
    train_cfg = json.loads((ROOT / 'stage1.json').read_text())
    train_cfg['soft_labels'] = 'cache/model_soft_labels.npy'
    train_cfg['soft_labels_sha256'] = manifest['soft_labels_sha256']
    train_cfg['soft_label_weight'] = args.weight
    train_cfg['prototype_contrast_weight'] = args.proto_weight
    config_path = ROOT / 'stage1_selftrain.json'
    config_path.write_text(json.dumps(train_cfg, indent=2) + '\n')

    print(json.dumps(dict(
        rows=int(len(labels)), soft_labels=str(out), config=str(config_path),
        seconds=round(time.monotonic() - began, 1),
        soft_labels_sha256=manifest['soft_labels_sha256'],
    )))


if __name__ == '__main__':
    main()
