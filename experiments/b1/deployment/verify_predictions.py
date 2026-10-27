import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1] / 'code'
names = json.loads((ROOT / 'data/a100/test_names.json').read_text())['names']
with Path(sys.argv[1]).open(newline='') as f:
    rows = list(csv.reader(f))
assert len(rows) == len(names) == 49857
assert [r[0] for r in rows] == names
assert len(set(names)) == len(names)
assert all(len(r) == 2 and len(r[1]) == 4 and r[1].isdigit() and 0 <= int(r[1]) < 500 for r in rows)
print(json.dumps({'verified_rows': len(rows), 'class_range': [0, 499], 'missing': 0, 'header': False}))
