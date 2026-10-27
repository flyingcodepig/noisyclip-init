import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import json
import math
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.attention import sdpa_kernel, SDPBackend
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.data.image_cache import CachedImageDataset, collate_cached
from single_gpu_support import load_model, load_actual_quality
from data_stream import StreamDataset, stream_collate
from trust_training import (configure_parameters, load_quality, classification_loss,
                            consistency_loss, freeze_warmup_gradients)
from infer import sha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--until', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart-from', help='B1 stage transition; optimizer preservation follows config')
    args = parser.parse_args()
    visible = [int(v) for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert visible and len(set(visible)) == len(visible)
    rank, world = (int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)))
    if world != 1:
        raise ValueError('This isolated trainer is single-GPU only')
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(Path(args.config).read_text())
    if (cfg['batch'] < 1 or cfg['total_steps'] < 1 or args.until < 1
            or cfg['head_warmup_steps'] < 0 or not 0 < cfg['layer_lr_decay'] <= 1
            or cfg['consistency'] < 0 or not 0 <= cfg['trusted_threshold'] <= 1):
        raise ValueError('Invalid B1 training settings')
    accum = cfg['gradient_accumulation_steps']
    if not isinstance(accum, int) or accum < 1:
        raise ValueError('Invalid accumulation')
    if args.until > cfg['total_steps']:
        raise ValueError('Target exceeds scheduler budget')
    ws = Path(args.workspace).resolve()
    assert ws == ROOT
    out = ws / 'runs' / args.run
    if not out.resolve().is_relative_to(ws / 'runs'):
        raise ValueError('Run path escapes runs directory')
    if not args.resume and (out / 'last.pt').exists():
        raise FileExistsError('Run already exists; use --resume or a new --run name')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = load_model(ws, cfg, ROOT).cuda()
    configure_parameters(model, cfg['unfreeze_last_blocks'])
    quality, quality_hash = load_actual_quality(ROOT, sha, 'cuda', cfg['num_classes'])
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(np.load(ROOT / 'cache/a100/initial_centers.npy')).cuda())
    anchors = {name: parameter.detach().clone() for name, parameter in model.backbone.named_parameters() if parameter.requires_grad and (not any((key in name for key in ('theta', 'gain', 'shift'))))}
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = 'head' if name.startswith('head.') or name == 'log_scale' else 'mechanism' if name in ('gate', 'log_temperature') else 'backbone'
        multiplier = 1.0
        if '.encoder.layers.' in name:
            layer = int(name.split('.encoder.layers.')[1].split('.')[0])
            depth = len(model.backbone.vision_model.encoder.layers) - 1 - layer
            multiplier = cfg['layer_lr_decay'] ** depth
        groups.append(dict(params=[parameter], lr=cfg['lr_' + key] * multiplier,
                           name=key, parameter_name=name, multiplier=multiplier,
                           initial_lr=cfg['lr_' + key] * multiplier))
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['wd'])
    covered = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(covered) == len(set(covered)) == sum((p.requires_grad for p in model.parameters()))
    start = 0
    stream_states = {}
    assert not (args.resume and args.restart_from)
    if args.restart_from:
        source = Path(args.restart_from).resolve()
        assert source.is_relative_to(ws), "Warm starts must use this workspace's own weights"
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        assert checkpoint.get('quality_sha256') == quality_hash
        assert checkpoint['config']['unfreeze_last_blocks'] == cfg['unfreeze_last_blocks']
        assert checkpoint['config']['layer_lr_decay'] == cfg['layer_lr_decay']
        model.load_state_dict(checkpoint['model'], strict=True)
        assert checkpoint['config']['size'] == cfg['size']
        preserve_optimizer = cfg.get('preserve_optimizer', False)
        if preserve_optimizer:
            assert 'optimizer' in checkpoint and checkpoint.get('reference_used_for_training') is False
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['initial_lr'] = cfg['lr_' + group['name']] * group['multiplier']
                group['weight_decay'] = cfg['wd']
        if rank == 0:
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (out / 'warm_start.json').write_text(json.dumps(dict(checkpoint=str(source), checkpoint_sha256=digest, previous_step=checkpoint['step'], optimizer_reset=not preserve_optimizer, augmentation_stream_reset=True, original_anchor_retained=True, resumed_world_size=world, previous_world_size=checkpoint.get('world_size')), indent=2))
    if args.resume:
        checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['config'] == cfg and checkpoint['world_size'] == world
        assert checkpoint.get('quality_sha256') == quality_hash
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        assert checkpoint['data_micro_step'] == start * accum
        assert checkpoint['accumulation_boundary'] is True
        rng = checkpoint['rng'][rank]
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        stream_states = rng.get('stream_states', {})
    teacher = torch.from_numpy(np.load(ROOT / 'cache/a100/train_teacher.npy')).cuda()
    labels = np.load(ROOT / 'cache/a100/train_labels.npy')
    dataset_class = StreamDataset if cfg.get('augmentation_mode') == 'worker_stream' else CachedImageDataset
    dataset_extra = dict(stream_states=stream_states, workers=cfg['workers'], alternate_cache=Path(cfg['image_cache_dir']) / 'semi_train_complete_512.npy' if cfg.get('alternate_complete_probability', 0) else None, alternate_probability=cfg.get('alternate_complete_probability', 0)) if dataset_class is StreamDataset else {}
    training_cache = Path(cfg['image_cache_dir']) / ('semi_train_complete_512.npy' if cfg.get('cache_kind') == 'complete' else 'semi_train_512.npy')
    ds = dataset_class(training_cache, np.arange(len(labels)), labels.tolist(), size=cfg['size'], train=True, scale=(cfg['rrc_min'], 1), wideshot_p=cfg['wideshot_p'], wideshot_scale=cfg['wideshot_scale'], wideshot_feather=cfg['feather'], jitter=cfg['jitter'], randaug_n=cfg.get('randaug_n', 0), randaug_m=cfg.get('randaug_m', 9), dual_view=True, seed_mode='worker' if dataset_class is StreamDataset else 'hash', base_seed=cfg['seed'] + rank * 10000019, **dataset_extra)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, seed=cfg['seed'], drop_last=True)
    if len(sampler) < cfg['batch']:
        raise ValueError('Not enough samples for one full batch per rank')
    net = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False) if world > 1 else model
    # Drop the epoch tail to a whole optimizer update; never save partial gradients.
    micro_per_epoch = (len(sampler) // cfg['batch'] // accum) * accum
    if micro_per_epoch == 0:
        raise ValueError('Not enough samples for one complete accumulated update')
    step, epoch = start, (start * accum) // micro_per_epoch
    update_losses = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rolling = []
    if rank == 0:
        (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        print(json.dumps(dict(stage='started', step=start, target=args.until, gpu=visible, trainable=sum((p.numel() for p in model.parameters() if p.requires_grad)), optimizer_coverage=True, official_initialization=True)), flush=True)

    def save():
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), stream_states=stream_states)
        rngs = [None] * world
        if world > 1:
            torch.distributed.all_gather_object(rngs, rng)
        else:
            rngs[0] = rng
        if rank == 0:
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, config=cfg, world_size=world, rng=rngs, official_initialization=True, reference_used_for_training=False, quality_sha256=quality_hash, data_micro_step=step * accum, accumulation_boundary=True)
            torch.save(state, out / 'last.tmp.pt')
            (out / 'last.tmp.pt').replace(out / 'last.pt')
            torch.save(dict(model=state['model'], config=cfg, step=step, quality_sha256=quality_hash), out / 'model.tmp.pt')
            (out / 'model.tmp.pt').replace(out / 'model.pt')
    while step < args.until:
        sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        steps_per_epoch = micro_per_epoch
        skip = start * accum % micro_per_epoch if epoch == start * accum // micro_per_epoch else 0
        if isinstance(ds, StreamDataset):
            ds._rng = None
            ds.stream_states = stream_states.copy()
            ds.stream_offset = skip % max(1, cfg['workers'])
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch * 179 + rank)
        selected = list(iter(sampler))[skip * cfg['batch']:steps_per_epoch * cfg['batch']]
        loader = DataLoader(ds, batch_size=cfg['batch'], sampler=selected, num_workers=cfg['workers'], collate_fn=stream_collate if isinstance(ds, StreamDataset) else collate_cached, pin_memory=True, drop_last=True, generator=generator, persistent_workers=False)
        for batch_no, batch in enumerate(loader):
            if step >= args.until:
                break
            stream_states.update(batch.get('stream_states', {}))
            images, y, index = (batch['image'], batch['label'], batch['index'])
            images, y, index = (images.cuda(non_blocking=True), y.cuda(non_blocking=True), index.cuda(non_blocking=True))
            warmup = max(1, int(cfg['total_steps'] * cfg['warmup_fraction']))
            fraction = (step + 1) / warmup if step < warmup else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, cfg['total_steps'] - warmup))))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * fraction
            if batch_no % accum == 0:
                optimizer.zero_grad(set_to_none=True)
                update_started = time.monotonic()
            perm = torch.randperm(len(y), device=y.device)
            mixed = random.random() < cfg['mix_probability']
            coefficient = 1.0
            cutmixed = False
            if mixed:
                cutmixed = random.random() < cfg.get('cutmix_probability', 0.0)
                a = cfg['cutmix_alpha'] if cutmixed else cfg['mix_alpha']
                coefficient = float(np.random.beta(a, a))
                if cutmixed:
                    from augment_mix import cutmix
                    images, coefficient, box = cutmix(images, perm, coefficient)
                    mixed = coefficient < 1.0
                else:
                    images = coefficient * images + (1 - coefficient) * images[perm]
            second_images = batch['image2'].cuda(non_blocking=True)
            forward_images = images if mixed else torch.cat((images, second_images), 0)
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                all_logits, all_features = net(forward_images)
            logits, features = all_logits[:len(y)].float(), all_features[:len(y)]
            consistency = logits.new_zeros(()) if mixed else consistency_loss(
                logits, all_logits[len(y):], quality['trust'][index], quality['group_weight'][index])
            cls_loss = classification_loss(
                logits, quality['targets'][index], quality['trust'][index],
                quality['group_weight'][index], cfg['gce_q'], cfg['gce_ce_floor'],
                perm if mixed else None, coefficient)
            target = F.normalize(coefficient * teacher[index] + (1 - coefficient) * teacher[index[perm]], dim=-1) if mixed else teacher[index]
            if cfg['drift_kind'] == 'huber':
                drift = F.smooth_l1_loss(features, target, beta=0.1, reduction='none').sum(-1).mean()
            else:
                drift = (features - target).square().sum(-1).mean()
            anchor = sum(((p.float() - anchors[n]).square().sum() for n, p in model.backbone.named_parameters() if n in anchors)) / max(1, len(anchors))
            contrast = features.new_zeros(())
            contrast_stats = {}
            if not mixed:
                if cfg.get('global_supcon', False):
                    from trusted_contrast import global_trusted_contrast
                    contrast, contrast_stats = global_trusted_contrast(features, y, (logits.detach().argmax(-1) == y) & (quality['trust'][index] >= cfg['trusted_threshold']), cfg['supcon_temperature'])
                else:
                    sim = features @ features.T / cfg['supcon_temperature']
                    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
                    trusted = (logits.detach().argmax(-1) == y) & (quality['trust'][index] >= cfg['trusted_threshold'])
                    positive = (y[:, None] == y[None]) & ~eye & trusted[:, None] & trusted[None]
                    count = positive.sum(-1)
                    logp = sim.masked_fill(eye, -10000.0).log_softmax(-1)
                    if (count > 0).any():
                        contrast = (-(logp * positive).sum(-1) / count.clamp_min(1))[count > 0].mean()
            loss = cls_loss + cfg['consistency'] * consistency + cfg['drift'] * drift + cfg['l2sp'] * anchor + cfg['supcon'] * contrast
            assert torch.isfinite(loss), f'Nonfinite training loss at step {step}'
            (loss / accum).backward()
            update_losses.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach())])
            if (batch_no + 1) % accum:
                continue
            freeze_warmup_gradients(model, step, cfg['head_warmup_steps'])
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            assert torch.isfinite(norm), f'Nonfinite gradient at step {step}'
            optimizer.step()
            step += 1
            torch.cuda.synchronize()
            update_seconds = time.monotonic() - update_started
            rolling.append(np.mean(update_losses, axis=0).tolist() + [float(norm)])
            update_losses.clear()
            if rank == 0 and (step % cfg.get('log_every', 25) == 0 or step == args.until):
                average = np.mean(rolling[-25:], axis=0).tolist()
                state = dict(stage='training', step=step, target=args.until, epoch=epoch, mean_loss=average[0], classification=average[1], consistency=float(consistency.detach()), mean_trust=float(quality['trust'][index].mean()), head_only=step <= cfg['head_warmup_steps'], drift=average[2], anchor=average[3], gradient_norm=average[4], cutmix_active=cutmixed, mixed=mixed, mixing_coefficient=coefficient, seconds=time.monotonic() - started, seconds_per_step=(time.monotonic() - started) / max(1, step - start), gpu=visible)
                state.update(micro_batch=cfg['batch'], accumulation=accum, effective_batch=cfg['batch'] * accum, update_seconds=update_seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, learning_rates=[g['lr'] for g in optimizer.param_groups], data_micro_step=step * accum)
                (out / 'status.json').write_text(json.dumps(state, indent=2))
                with (out / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(state) + '\n')
                state.update(contrast_stats)
                print(json.dumps(state), flush=True)
            if step % cfg['save_every'] == 0 or step == args.until:
                save()
        epoch += 1
    if rank == 0:
        print(json.dumps(dict(stage='completed', step=step, seconds=time.monotonic() - started)), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()
if __name__ == '__main__':
    main()
