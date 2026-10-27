import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(ROOT))
from fetch_weights import digest
from engine.trust_quality import crossfit, make_targets

def main():
    out = ROOT / 'cache/a100/train_quality.npz'
    if out.exists():
        report = json.loads(out.with_suffix('.json').read_text())
        assert digest(out) == report['quality_sha256']
        for name, expected in report['inputs'].items():
            assert digest(ROOT / name) == expected, name
        print('Existing actual-data quality verified', flush=True)
        return
    began = time.monotonic()
    provenance = ROOT / 'cache/a100/feature_provenance.json'
    record = json.loads(provenance.read_text())
    binding = dict(record['hashes'])
    binding[str(provenance.relative_to(ROOT))] = digest(provenance)
    for name, expected in binding.items():
        assert digest(ROOT / name) == expected, name
    rows = json.loads((ROOT / 'data/a100/train_index.json').read_text())
    manifest = dict(json.loads((ROOT / 'data/a100/train_sha256.json').read_text()))
    groups = []
    data_root = Path(json.loads((ROOT / 'data/a100/source_manifest.json').read_text())['splits']['train']['root'])
    for i, (name, _) in enumerate(rows):
        value = digest(data_root / name)
        assert value == manifest[name], name
        groups.append(value)
        if i % 10000 == 0:
            print(json.dumps({'stage': 'verify_train', 'rows': i}), flush=True)
    groups = np.asarray(groups)
    labels = np.load(ROOT / 'cache/a100/train_labels.npy')
    features = np.load(ROOT / 'cache/a100/train_teacher.npy')
    assert np.array_equal(labels, [r[1] for r in rows])
    assert features.shape == (len(rows), 512) and np.isfinite(features).all()
    cfg = json.loads((ROOT / 'quality_config.json').read_text())
    cfg['device'] = 'cuda'
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    p, v, folds, missing = crossfit(features, labels, groups, cfg)
    # Every duplicate content group must belong to exactly one held-out fold.
    assigned = {}
    for g, f in zip(groups, folds):
        assert assigned.setdefault(g, int(f)) == int(f)
    soft, trust, weight = make_targets(p, v, labels, groups, cfg)
    assert np.isfinite(soft).all() and np.allclose(soft.sum(1), 1, atol=1e-5)
    temp = out.with_suffix('.building.npz')
    np.savez(temp, targets=soft, trust=trust, group_weight=weight, groups=groups, folds=folds, oof=p, neighbor_probability=v)
    temp.replace(out)
    report = {'config': cfg, 'inputs': binding, 'quality_sha256': digest(out), 'rows': len(rows), 'groups': len(assigned), 'missing_classes_by_fold': missing, 'duplicate_fold_isolation_verified': True, 'test_labels_read': False, 'trust_quantiles': np.quantile(trust, [0, .25, .5, .75, 1]).tolist(), 'seconds': time.monotonic() - began}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
