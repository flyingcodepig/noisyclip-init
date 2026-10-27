"""Real full-input GPU smoke: head warmup, unfreeze, exact split/resume comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import hashlib
import argparse

ROOT = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(ROOT / 'engine'))
from single_gpu_support import load_model

def equal_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b), 'Tensor differs after resume'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b, (a, b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int)
    parser.add_argument('--accum', type=int)
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prefix = 'smoke_' + time.strftime('%Y%m%d_%H%M%S')
    folder = ROOT / 'runs' / prefix
    folder.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'train_config_a100_stage1.json').read_text())
    if args.batch is not None or args.accum is not None:
        assert args.no_gate and args.batch and args.accum
        assert args.batch * args.accum == 192
        cfg.update(batch=args.batch, gradient_accumulation_steps=args.accum)
    cfg.update(total_steps=8, head_warmup_steps=2, save_every=2, log_every=1, mix_probability=0.0)
    config = folder / 'config.json'
    config.write_text(json.dumps(cfg, indent=2))
    base = load_model(ROOT, cfg, ROOT)
    initial = {k: v.detach().clone() for k, v in base.state_dict().items() if k.startswith('backbone.') or k in ['gate', 'log_temperature']}
    del base

    def run(name, until, resume=False):
        command = [sys.executable, str(ROOT / 'engine/train_single_gpu.py'), '--workspace', str(ROOT), '--config', str(config), '--run', f'{prefix}/{name}', '--until', str(until)]
        if resume:
            command.append('--resume')
        with (folder / f'{name}_{until}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return torch.load(folder / name / 'last.pt', map_location='cpu', weights_only=False)

    warm = run('split', 2)
    assert warm['step'] == 2
    for k, tensor in initial.items():
        assert torch.equal(tensor, warm['model'][k]), f'Warmup changed {k}'
    import numpy as np
    centers = torch.from_numpy(np.load(ROOT / 'cache/a100/initial_centers.npy'))
    assert not torch.equal(centers, warm['model']['head.weight']), 'Head did not update'
    del warm
    resumed = run('split', 4, True)
    for layer in range(12):
        keys = [k for k in initial if f'.encoder.layers.{layer}.' in k]
        changed = any(not torch.equal(initial[k], resumed['model'][k]) for k in keys)
        assert changed == (layer >= 8), f'Incorrect layer freeze: {layer}'
    uninterrupted = run('continuous', 4)
    equal_tree(resumed['model'], uninterrupted['model'])
    equal_tree(resumed['optimizer'], uninterrupted['optimizer'])
    equal_tree(resumed['rng'][0]['torch'], uninterrupted['rng'][0]['torch'])
    equal_tree(resumed['rng'][0]['cuda'], uninterrupted['rng'][0]['cuda'])
    equal_tree(resumed['rng'][0]['python'], uninterrupted['rng'][0]['python'])
    equal_tree(resumed['rng'][0]['stream_states'], uninterrupted['rng'][0]['stream_states'])
    assert np.array_equal(resumed['rng'][0]['numpy'][1], uninterrupted['rng'][0]['numpy'][1])
    assert resumed['data_micro_step'] == uninterrupted['data_micro_step'] == 4 * cfg['gradient_accumulation_steps']
    metrics = [json.loads(line) for line in (folder / 'continuous/metrics.jsonl').read_text().splitlines()]
    seconds = sum(m['update_seconds'] for m in metrics[2:]) / 2
    stage2 = json.loads((ROOT / 'train_config_a100_stage2.json').read_text())
    stage2.update(batch=cfg['batch'], gradient_accumulation_steps=cfg['gradient_accumulation_steps'], save_every=2, log_every=1)
    stage2_path = folder / 'stage2_config.json'
    stage2_path.write_text(json.dumps(stage2, indent=2))
    with (folder / 'stage2.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT/'engine/train_single_gpu.py'), '--workspace', str(ROOT), '--config', str(stage2_path), '--run', f'{prefix}/stage2', '--until', '2', '--restart-from', str(folder/'split/last.pt')], stdout=log, stderr=subprocess.STDOUT, check=True)
    transitioned = torch.load(folder/'stage2/last.pt', map_location='cpu', weights_only=False)
    assert transitioned['step'] == 2 and transitioned['config']['total_steps'] == 3072
    for k, state in transitioned['optimizer']['state'].items():
        assert state['step'] == resumed['optimizer']['state'][k]['step'] + 2
    for group in transitioned['optimizer']['param_groups']:
        expected = stage2['lr_' + group['name']] * group['multiplier'] * 2 / int(3072 * stage2['warmup_fraction'])
        assert abs(group['lr'] - expected) < 1e-12
    report = {'status': 'passed', 'folder': str(folder), 'full_quality_used': True, 'cuda_bf16_real_model': True, 'head_warmup_backbone_frozen': True, 'last_four_layers_updated': True, 'resume_model_optimizer_rng_exact': True, 'micro_batch': cfg['batch'], 'accumulation': cfg['gradient_accumulation_steps'], 'peak_allocated_gib': max(m['peak_allocated_gib'] for m in metrics), 'post_warmup_seconds_per_update': seconds, 'estimated_training_hours_6144_updates': seconds * 6144 / 3600, 'estimate_excludes_preparation_checkpoint_io_and_variability': True}
    (folder / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report['formal_config_hashes'] = {f'train_config_a100_{s}.json': digest(ROOT / f'train_config_a100_{s}.json') for s in ['stage1', 'stage2']}
    report['training_sources'] = resumed['training_sources']
    report['quality_sha256'] = resumed['quality_sha256']
    report['git_revision'] = resumed['git_revision']
    report['stage2_optimizer_preserved_schedule_restarted'] = True
    if not args.no_gate:
        (ROOT / 'cache/a100/smoke_gate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
