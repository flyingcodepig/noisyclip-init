"""Reproducibly generate the isolated A100 adaptation; never edit baseline/B1 originals."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'code'
source = (ROOT / 'engine/train_with_cutmix_four_gpu.py').read_text(encoding='utf-8')

def replace(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f'Unexpected upstream trainer: {old[:100]}')
    source = source.replace(old, new)

replace("    if world > 1:\n        torch.distributed.init_process_group('nccl')", "    if world != 1:\n        raise ValueError('This isolated trainer is single-GPU only')")
replace('from model_base import load_model', 'from single_gpu_support import load_model, load_actual_quality')
replace("quality, quality_hash = load_quality(ROOT, sha, 'cuda')", "quality, quality_hash = load_actual_quality(ROOT, sha, 'cuda', cfg['num_classes'])")
replace("    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()", "    teacher = torch.from_numpy(np.load(ROOT / 'cache/a100/train_teacher.npy')).cuda()")
replace("    labels = np.load(HERE / 'train_labels.npy')", "    labels = np.load(ROOT / 'cache/a100/train_labels.npy')")
replace("np.load(HERE / 'initial_centers.npy')", "np.load(ROOT / 'cache/a100/initial_centers.npy')")
replace("    ws = Path(args.workspace).resolve()", "    accum = cfg['gradient_accumulation_steps']\n    if not isinstance(accum, int) or accum < 1:\n        raise ValueError('Invalid accumulation')\n    if args.until > cfg['total_steps']:\n        raise ValueError('Target exceeds scheduler budget')\n    ws = Path(args.workspace).resolve()")
replace("        start = checkpoint['step']", "        start = checkpoint['step']\n        assert checkpoint['data_micro_step'] == start * accum\n        assert checkpoint['accumulation_boundary'] is True")
replace("    step, epoch = (start, start // (len(sampler) // cfg['batch']))", "    # Drop the epoch tail to a whole optimizer update; never save partial gradients.\n    micro_per_epoch = (len(sampler) // cfg['batch'] // accum) * accum\n    if micro_per_epoch == 0:\n        raise ValueError('Not enough samples for one complete accumulated update')\n    step, epoch = start, (start * accum) // micro_per_epoch\n    update_losses = []\n    torch.cuda.reset_peak_memory_stats()")
replace("quality_sha256=quality_hash)\n            torch.save(state", "quality_sha256=quality_hash, data_micro_step=step * accum, accumulation_boundary=True)\n            torch.save(state")
replace("        steps_per_epoch = len(sampler) // cfg['batch']\n        skip = start % steps_per_epoch if epoch == start // steps_per_epoch else 0", "        steps_per_epoch = micro_per_epoch\n        skip = start * accum % micro_per_epoch if epoch == start * accum // micro_per_epoch else 0")
replace("            optimizer.zero_grad(set_to_none=True)", "            if batch_no % accum == 0:\n                optimizer.zero_grad(set_to_none=True)\n                update_started = time.monotonic()")
replace("            loss.backward()\n            freeze_warmup_gradients", "            (loss / accum).backward()\n            update_losses.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach())])\n            if (batch_no + 1) % accum:\n                continue\n            freeze_warmup_gradients")
replace("            rolling.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach()), float(norm)])", "            torch.cuda.synchronize()\n            update_seconds = time.monotonic() - update_started\n            rolling.append(np.mean(update_losses, axis=0).tolist() + [float(norm)])\n            update_losses.clear()")
replace("(step % 25 == 0 or step == args.until)", "(step % cfg.get('log_every', 25) == 0 or step == args.until)")
replace("                (out / 'status.json').write_text", "                state.update(micro_batch=cfg['batch'], accumulation=accum, effective_batch=cfg['batch'] * accum, update_seconds=update_seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, learning_rates=[g['lr'] for g in optimizer.param_groups], data_micro_step=step * accum)\n                (out / 'status.json').write_text")
(ROOT / 'engine/train_single_gpu.py').write_text(source, encoding='utf-8')
for stage, filename in [('stage1', 'train_config_stage1.json'), ('stage2', 'train_config.json')]:
    cfg = json.loads((ROOT / filename).read_text())
    cfg.update(batch=24, gradient_accumulation_steps=8, workers=6, log_every=5, num_classes=500, dataset_id='matpool_semifinal_20261027')
    (ROOT / f'train_config_a100_{stage}.json').write_text(json.dumps(cfg, indent=2) + '\n')

cache = (ROOT / 'build_cache.py').read_text(encoding='utf-8')
cache = cache.replace('data/train_index.json', 'data/a100/train_index.json').replace('data/test_names.json', 'data/a100/test_names.json')
(ROOT / 'build_cache_a100.py').write_text(cache, encoding='utf-8')

