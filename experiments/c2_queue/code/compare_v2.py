"""Compare stage1_v2 (improved) against stage1 (baseline C2) on held-out validation."""
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

a = read(ROOT / 'runs/stage1/validation_predictions.csv')       # baseline
b = read(ROOT / 'runs/stage1_v2/validation_predictions.csv')    # improved
assert len(a) == len(b) == 9020
assert [(r['filename'], r['given_label']) for r in a] == [(r['filename'], r['given_label']) for r in b]
old = np.array([r['given_label'] == r['prediction'] for r in a])
new = np.array([r['given_label'] == r['prediction'] for r in b])
ra = json.loads((ROOT / 'runs/stage1/validation.json').read_text())
rb = json.loads((ROOT / 'runs/stage1_v2/validation.json').read_text())
out = dict(
    baseline_acc=ra['accuracy'], baseline_macro=ra['macro_accuracy'],
    improved_acc=rb['accuracy'], improved_macro=rb['macro_accuracy'],
    accuracy_delta_pp=100 * float(new.mean() - old.mean()),
    macro_delta_pp=100 * float(rb['macro_accuracy'] - ra['macro_accuracy']),
    corrected=int((~old & new).sum()), regressed=int((old & ~new).sum()),
    both_correct=int((old & new).sum()), both_wrong=int((~old & ~new).sum()),
    changed_predictions=int((np.array([r['prediction'] for r in a]) != np.array([r['prediction'] for r in b])).sum()),
    rows=len(a),
)
print(json.dumps(out, indent=2))
(ROOT / 'runs/comparison_v2.json').write_text(json.dumps(out, indent=2) + '\n')
