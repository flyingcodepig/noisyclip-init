"""Fail closed on incompatible/incomplete B1 checkpoints."""
import hashlib
import json
import sys
from pathlib import Path
import torch

path, config, target, root = sys.argv[1:]
cfg = json.loads(Path(config).read_text())
ck = torch.load(path, map_location='cpu', weights_only=False)
assert ck['config'] == cfg
assert ck['world_size'] == 1
assert int(target) <= ck['step'] <= cfg['total_steps']
assert ck['data_micro_step'] == ck['step'] * cfg['gradient_accumulation_steps']
assert ck['accumulation_boundary'] is True
assert ck['reference_used_for_training'] is False
quality = Path(root) / 'cache/a100/train_quality.npz'
h = hashlib.sha256()
with quality.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
assert ck['quality_sha256'] == h.hexdigest()
assert ck['optimizer']['state']
assert all(torch.isfinite(t).all() for t in ck['model'].values() if t.is_floating_point())
for state in ck['optimizer']['state'].values():
    assert all(torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t))
assert len(ck['rng']) == 1 and ck['rng'][0]['stream_states']
print(json.dumps({'checkpoint_verified': path, 'step': ck['step'], 'quality': h.hexdigest()}))
