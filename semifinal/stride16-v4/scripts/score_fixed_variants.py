import csv,json
from pathlib import Path
r=Path('/root/aic_a100/eat_semifinal')
truth=json.loads((r/'split/validation_labels.json').read_text())
for name in ('fixed_raw','fixed_power','fixed_w075','fixed_w125','fixed_t008','fixed_t012'):
    path=r/'runs/dense'/name/'pred_results.csv'
    with path.open(newline='') as f: rows=dict(csv.reader(f))
    assert len(rows)==len(truth) and set(rows)==set(truth)
    n=sum(rows[k]==v for k,v in truth.items())
    print(name,n,len(truth),n/len(truth),flush=True)
