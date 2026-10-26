"""Build reproducible, TRAIN-only B1 supervision from frozen official features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from engine.trust_quality import crossfit, make_targets

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'quality_config.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--force', action='store_true', help='Explicitly replace quality artifacts; existing runs will reject a changed hash')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / 'cache/train_quality.npz'
    if out.exists() and not args.force:
        raise FileExistsError(f'{out}: preserve existing runs; use --force only for an intentional rebuild')
    if (cfg['ridge'] <= 0 or cfg['temperature'] <= 0 or cfg['neighbor_temperature'] <= 0
            or cfg['neighbors'] < 1 or cfg['batch'] < 1
            or not 0 < cfg['minimum_trust'] <= 1
            or not 0 <= cfg['correction_threshold'] <= 1):
        raise ValueError('Invalid quality configuration')
    if args.device:
        cfg['device'] = args.device
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads((ROOT / 'data/train_index.json').read_text())
    recipe = json.loads((ROOT / 'training_cache_recipe.json').read_text())['expected_sha256']
    inputs = ['data/train_index.json', 'engine/train_teacher.npy', 'engine/train_labels.npy']
    binding = {name: digest(ROOT / name) for name in inputs}
    for name in inputs[1:]:
        if binding[name] != recipe[name]:
            raise ValueError(f'Official feature/label provenance mismatch: {name}')
    labels = np.load(ROOT / 'engine/train_labels.npy', allow_pickle=False)
    features = np.load(ROOT / 'engine/train_teacher.npy', allow_pickle=False)
    if not np.array_equal(labels, np.asarray([r[1] for r in rows])):
        raise ValueError('TRAIN index order differs from labels')
    if features.shape != (len(rows), 512) or not np.isfinite(features).all():
        raise ValueError('Invalid frozen features')
    groups = []
    root = (ROOT / 'data/official/train').resolve()
    for i, (name, _) in enumerate(rows):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('TRAIN path escapes root')
        groups.append(digest(path))
        if i % 10000 == 0:
            print(json.dumps({'stage': 'hash_train', 'rows': i}), flush=True)
    groups = np.asarray(groups)
    probability, neighbors, fold, missing = crossfit(features, labels, groups, cfg)
    soft, trust, weight = make_targets(probability, neighbors, labels, groups, cfg)
    out.parent.mkdir(exist_ok=True)
    temporary = out.with_suffix('.building.npz')
    np.savez(temporary, targets=soft, trust=trust, group_weight=weight,
             folds=fold, groups=groups, oof=probability, neighbor_probability=neighbors)
    temporary.replace(out)
    report = {'config': cfg, 'inputs': binding, 'quality_sha256': digest(out),
              'rows': len(rows), 'groups': len(np.unique(groups)),
              'missing_classes_by_fold': missing,
              'trust_quantiles': np.quantile(trust, [0, .25, .5, .75, 1]).tolist(),
              'classes': {str(c): {'rows': int((labels == c).sum()),
                                   'mean_trust': float(trust[labels == c].mean())}
                          for c in np.unique(labels)}}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stage': 'quality_complete', 'path': str(out), 'groups': report['groups']}))


if __name__ == '__main__':
    main()
