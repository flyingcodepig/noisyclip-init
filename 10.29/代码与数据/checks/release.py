"""Validate the supplied checkpoint, submitted result and optional reproduced CSV."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ops.runtime import ENTRY, file_hash, state_hash, FEATURE_SHA, MANIFEST_SHA

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path)
    args = parser.parse_args()
    official = json.loads((ROOT / 'checks/official_result.json').read_text())
    checkpoint = ROOT.parent / (ENTRY + '-模型文件.pt')
    if file_hash(checkpoint) != official['model_sha256']:
        raise ValueError('Supplied checkpoint checksum differs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state_hash(payload['weights']) != official['tensor_state_sha256']:
        raise ValueError('Model tensor checksum differs')
    if payload['routing_strength'] != 0. or payload['selected_epoch'] != 14 or len(payload['weights']) != 205:
        raise ValueError('Unexpected model configuration')
    archive = ROOT.parent / (ENTRY + '-赛题结果文档.zip')
    if file_hash(archive) != official['submission_zip_sha256']:
        raise ValueError('Submitted ZIP differs')
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ['pred_results.csv']:
            raise ValueError('ZIP must contain only pred_results.csv')
        raw = z.read('pred_results.csv')
    if hashlib.sha256(raw).hexdigest() != official['prediction_sha256']:
        raise ValueError('Submitted CSV checksum differs')
    names = json.loads((ROOT / 'inputs/test_names.json').read_text())
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'))))
    if len(rows) != 49857 or [r[0] for r in rows] != names or any(len(r) != 2 or len(r[1]) != 4 or not r[1].isdigit() or int(r[1]) >= 500 for r in rows):
        raise ValueError('Submitted CSV format differs')
    if file_hash(ROOT / 'inputs/official_clip224_features.npy') != FEATURE_SHA:
        raise ValueError('Official TRAIN features changed')
    if file_hash(ROOT / 'inputs/train_manifest.jsonl') != MANIFEST_SHA:
        raise ValueError('TRAIN manifest changed')
    if args.prediction and args.prediction.read_bytes() != raw:
        raise ValueError('The reproduced prediction differs from the submitted CSV')
    print(json.dumps(dict(status='passed', model_tensors=205, TEST_images=49857,
        prediction_checked=bool(args.prediction), score=official['score']), ensure_ascii=False))

if __name__ == '__main__':
    main()
