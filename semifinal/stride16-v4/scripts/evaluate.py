import csv,json,hashlib
from pathlib import Path
root=Path('/root/aic_a100/eat_semifinal')
prediction=root/'runs/baseline/validation/pred_results.csv'
with prediction.open() as f: rows=list(csv.reader(f))
pred=dict(rows); truth=json.loads((root/'split/validation_labels.json').read_text())
assert len(pred)==len(rows)==len(truth) and set(pred)==set(truth)
correct=sum(pred[n]==y for n,y in truth.items())
report={'metric':'held_out_noisy_label_accuracy_not_official_score','correct':correct,
        'total':len(truth),'accuracy':correct/len(truth),
        'prediction_sha256':hashlib.sha256(prediction.read_bytes()).hexdigest(),
        'split_sha256':hashlib.sha256((root/'split/manifest.json').read_bytes()).hexdigest()}
(root/'runs/baseline/validation_score.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report)); print('SCORE:',100*report['accuracy'])
