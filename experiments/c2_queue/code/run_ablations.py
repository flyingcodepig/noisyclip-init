"""Run CBD/EMA hyperparameter ablations sequentially on a single GPU.

Each ablation reuses the base stage1.json and overrides only the
class-balanced / EMA keys, then trains to 4096 updates and evaluates on the
held-out noisy TRAIN split (never the competition TEST set). Results are
written to runs/ablation_results.json; consensus scoring is a separate step
(score_local.py) run after all ablations finish.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = '/root/b1-env-20261027/bin/python'

# (name, overrides). A None value removes the key; anything else replaces it.
ABLATIONS = [
    ('ab_no_cbd',    {'class_balanced_beta': None}),
    ('ab_no_ema',    {'ema_decay': None, 'ema_drift_weight': None}),
    ('ab_beta_9999', {'class_balanced_beta': 0.9999}),
    ('ab_beta_995',  {'class_balanced_beta': 0.995}),
    ('ab_ema_001',   {'ema_drift_weight': 0.01}),
    ('ab_ema_004',   {'ema_drift_weight': 0.04}),
]


def log(message):
    print(json.dumps(message), flush=True)


def main():
    base = json.loads((ROOT / 'stage1.json').read_text())
    summary = {}
    log({'event': 'ablation_start',
         'base_class_balanced_beta': base.get('class_balanced_beta'),
         'base_ema_decay': base.get('ema_decay'),
         'base_ema_drift_weight': base.get('ema_drift_weight'),
         'ablation_count': len(ABLATIONS)})
    for name, overrides in ABLATIONS:
        started = time.monotonic()
        run_dir = ROOT / 'runs' / name
        if (run_dir / 'validation.json').exists():
            report = json.loads((run_dir / 'validation.json').read_text())
            summary[name] = {'accuracy': report['accuracy'],
                             'macro_accuracy': report['macro_accuracy'],
                             'status': 'skipped_existing'}
            log({'event': 'skip_existing', 'run': name,
                 'accuracy': report['accuracy']})
            continue
        cfg = dict(base)
        for key, value in overrides.items():
            if value is None:
                cfg.pop(key, None)
            else:
                cfg[key] = value
        config_path = ROOT / f'ablation_{name}.json'
        config_path.write_text(json.dumps(cfg, indent=2) + '\n')
        log({'event': 'train_start', 'run': name, 'overrides': overrides})
        try:
            subprocess.run(
                [PYTHON, 'engine/train_stage1.py', '--workspace', str(ROOT),
                 '--config', str(config_path), '--run', name, '--until', '4096'],
                cwd=ROOT, check=True)
            log({'event': 'train_done', 'run': name,
                 'seconds': round(time.monotonic() - started, 1)})
            subprocess.run([PYTHON, 'evaluate.py', '--stage', name],
                           cwd=ROOT, check=True)
            report = json.loads((run_dir / 'validation.json').read_text())
            summary[name] = {'accuracy': report['accuracy'],
                             'macro_accuracy': report['macro_accuracy'],
                             'seconds': round(time.monotonic() - started, 1),
                             'status': 'ok'}
            log({'event': 'eval_done', 'run': name,
                 'accuracy': report['accuracy'],
                 'macro_accuracy': report['macro_accuracy']})
        except subprocess.CalledProcessError as exc:
            summary[name] = {'status': 'error', 'returncode': exc.returncode,
                             'seconds': round(time.monotonic() - started, 1)}
            log({'event': 'error', 'run': name, 'returncode': exc.returncode})
    (ROOT / 'runs' / 'ablation_results.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    log({'event': 'ablation_done', 'summary': summary})


if __name__ == '__main__':
    main()
