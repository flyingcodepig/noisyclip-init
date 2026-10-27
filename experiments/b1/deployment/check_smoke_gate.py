import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / 'code'
report = json.loads((ROOT / 'cache/a100/smoke_gate.json').read_text())
assert report['status'] == 'passed' and report['resume_model_optimizer_rng_exact']
for name, expected in {**report['formal_config_hashes'], **report['training_sources']}.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
quality = json.loads((ROOT / 'cache/a100/train_quality.json').read_text())
assert report['quality_sha256'] == quality['quality_sha256']
print('Full-input smoke gate verified')
