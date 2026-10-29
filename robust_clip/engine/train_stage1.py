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
from model_base import load_model
from data_stream import StreamDataset, stream_collate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--until', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--restart-from', help='Own-workspace weights, with a fresh optimizer and augmentation stream')
    args = parser.parse_args()
    visible = [int(v) for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert visible and len(set(visible)) == len(visible) and all((v in (0, 1, 2, 3) for v in visible))
    rank, world = (int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)))
    if world > 1:
        torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(Path(args.config).read_text())
    ws = Path(args.workspace).resolve()
    assert ws == ROOT
    out = ws / 'runs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = load_model(ws, cfg, ROOT).cuda()
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(np.load(HERE / 'initial_centers.npy')).cuda())
    anchors = {name: parameter.detach().clone() for name, parameter in model.backbone.named_parameters() if parameter.requires_grad and (not any((key in name for key in ('theta', 'gain', 'shift'))))}
    groups = {'backbone': [], 'mechanism': [], 'head': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = 'head' if name.startswith('head.') or name == 'log_scale' else 'mechanism' if any((token in name for token in ('theta', 'gain', 'shift', 'gate', 'log_temperature'))) else 'backbone'
        groups[key].append(parameter)
    optimizer = torch.optim.AdamW([dict(params=params, lr=cfg['lr_' + key], name=key, initial_lr=cfg['lr_' + key]) for key, params in groups.items() if params], weight_decay=cfg['wd'])
    covered = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(covered) == len(set(covered)) == sum((p.requires_grad for p in model.parameters()))
    start = 0
    stream_states = {}
    assert not (args.resume and args.restart_from)
    if args.restart_from:
        source = Path(args.restart_from).resolve()
        assert source.is_relative_to(ws), "Warm starts must use this workspace's own weights"
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        assert checkpoint['config']['size'] == cfg['size']
        preserve_optimizer = cfg.get('preserve_optimizer', False)
        if preserve_optimizer:
            assert 'optimizer' in checkpoint and checkpoint.get('reference_used_for_training') is False
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['initial_lr'] = cfg['lr_' + group['name']]
                group['weight_decay'] = cfg['wd']
        if rank == 0:
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (out / 'warm_start.json').write_text(json.dumps(dict(checkpoint=str(source), checkpoint_sha256=digest, previous_step=checkpoint['step'], optimizer_reset=not preserve_optimizer, augmentation_stream_reset=True, original_anchor_retained=True, resumed_world_size=world, previous_world_size=checkpoint.get('world_size')), indent=2))
    if args.resume:
        checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['config'] == cfg and checkpoint['world_size'] == world
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        rng = checkpoint['rng'][rank]
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        stream_states = rng.get('stream_states', {})
    teacher = torch.from_numpy(np.load(HERE / 'train_teacher.npy')).cuda()
    labels = np.load(HERE / 'train_labels.npy')
    own_features, own_probability = (None, None)
    if cfg.get('own_teacher'):
        from infer import sha
        teacher_path = (ws / cfg['own_teacher']).resolve()
        binding = json.loads(teacher_path.with_suffix('.json').read_text())
        assert teacher_path.is_relative_to(ws) and binding['reference_read'] is False
        assert sha(teacher_path) == cfg['own_teacher_sha256'] == binding['teacher_sha256']
        with np.load(teacher_path) as data:
            own_features = torch.from_numpy(data['features']).cuda()
            own_probability = torch.from_numpy(data['probabilities']).cuda()
        assert own_features.shape == teacher.shape and own_probability.shape == (len(labels), 750)
    sample_weights = None
    if cfg.get('loss_weights'):
        assert ws == ROOT
        from semantic_supervision import weighted_loss
        from infer import sha
        weight_path = (ws / cfg['loss_weights']).resolve()
        assert weight_path.is_relative_to(ws)
        binding = json.loads(weight_path.with_suffix('.json').read_text())
        assert binding['reference_read'] is False and binding['original_label_overwritten'] is False
        assert sha(weight_path) == cfg['loss_weights_sha256'] == binding['weights_sha256']
        assert sha(HERE / 'train_teacher.npy') == binding['teacher_sha256']
        assert sha(HERE / 'train_labels.npy') == binding['labels_sha256']
        sample_weights = torch.from_numpy(np.load(weight_path)).cuda()
        assert sample_weights.shape == labels.shape and torch.isfinite(sample_weights).all()
    candidate_labels = None
    if cfg.get('candidate_labels'):
        from infer import sha
        candidate_path = (ws / cfg['candidate_labels']).resolve()
        assert candidate_path.is_relative_to(ws) and sha(candidate_path) == cfg['candidate_labels_sha256']
        binding = json.loads(candidate_path.with_suffix('.json').read_text())
        assert binding['reference_read'] is False and binding['labels_order_verified'] is True
        candidate_labels = torch.from_numpy(np.load(candidate_path)).cuda()
        assert len(candidate_labels) == len(labels)
        assert (candidate_labels == torch.from_numpy(labels).cuda()[:, None]).any(-1).all()
    accumulation = int(cfg.get('gradient_accumulation', 1))
    assert accumulation >= 1
    if cfg.get('data_backend') == 'zip':
        from semi_data import RawStreamDataset
        ds = RawStreamDataset(cfg, stream_states)
    else:
        ds = None
    dataset_class = StreamDataset if cfg.get('augmentation_mode') == 'worker_stream' else CachedImageDataset
    dataset_extra = dict(stream_states=stream_states, workers=cfg['workers'], alternate_cache=ROOT / 'image_cache/semi_train_complete_512.npy' if cfg.get('alternate_complete_probability', 0) else None, alternate_probability=cfg.get('alternate_complete_probability', 0)) if dataset_class is StreamDataset else {}
    training_cache = ROOT / ('image_cache/semi_train_complete_512.npy' if cfg.get('cache_kind') == 'complete' else 'image_cache/semi_train_512.npy')
    if ds is None:
        ds = dataset_class(training_cache, np.arange(len(labels)), labels.tolist(), size=cfg['size'], train=True, scale=(cfg['rrc_min'], 1), wideshot_p=cfg['wideshot_p'], wideshot_scale=cfg['wideshot_scale'], wideshot_feather=cfg['feather'], jitter=cfg['jitter'], randaug_n=cfg.get('randaug_n', 0), randaug_m=cfg.get('randaug_m', 9), seed_mode='worker' if dataset_class is StreamDataset else 'hash', base_seed=cfg['seed'] + rank * 10000019, **dataset_extra)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, seed=cfg['seed'], drop_last=True)
    net = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False) if world > 1 else model
    steps_per_epoch = len(sampler) // (cfg['batch'] * accumulation)
    assert steps_per_epoch > 0
    step, epoch = (start, start // steps_per_epoch)
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
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, config=cfg, world_size=world, rng=rngs, official_initialization=True, reference_used_for_training=False)
            torch.save(state, out / 'last.tmp.pt')
            (out / 'last.tmp.pt').replace(out / 'last.pt')
            torch.save(dict(model=state['model'], config=cfg, step=step), out / 'model.tmp.pt')
            (out / 'model.tmp.pt').replace(out / 'model.pt')
    while step < args.until:
        sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        skip = start % steps_per_epoch if epoch == start // steps_per_epoch else 0
        if isinstance(ds, StreamDataset):
            ds._rng = None
            ds.stream_states = stream_states.copy()
            ds.stream_offset = (skip * accumulation) % max(1, cfg['workers'])
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch * 179 + rank)
        selected = list(iter(sampler))[skip * cfg['batch'] * accumulation:steps_per_epoch * cfg['batch'] * accumulation]
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
            if batch_no % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
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
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                logits, features = net(images)
            logits = logits.float()
            if mixed:
                if candidate_labels is None:
                    cls_loss = coefficient * F.cross_entropy(logits, y) + (1 - coefficient) * F.cross_entropy(logits, y[perm])
                else:
                    choices = candidate_labels[index]
                    logp = logits.log_softmax(-1)
                    partial_ce = -logp.gather(1, choices.clamp_min(0)).masked_fill(choices < 0, -torch.inf).logsumexp(-1)
                    second = choices[perm]
                    secondary_ce = -logp.gather(1, second.clamp_min(0)).masked_fill(second < 0, -torch.inf).logsumexp(-1)
                    cls_loss = (coefficient * partial_ce + (1 - coefficient) * secondary_ce).mean()
                target = F.normalize(coefficient * teacher[index] + (1 - coefficient) * teacher[index[perm]], dim=-1)
            else:
                log_probability = logits.log_softmax(-1).gather(1, y[:, None]).squeeze(1)
                if candidate_labels is not None:
                    choices = candidate_labels[index]
                    log_probability = logits.log_softmax(-1).gather(1, choices.clamp_min(0)).masked_fill(choices < 0, -torch.inf).logsumexp(-1)
                cls_loss = ((1 - (cfg['gce_q'] * log_probability).exp()) / cfg['gce_q']).mean()
                floor = cfg.get('gce_ce_floor', 0.0)
                assert 0 <= floor <= 1
                cls_loss = (1 - floor) * cls_loss + floor * F.cross_entropy(logits, y)
                target = teacher[index]
            if sample_weights is not None:
                cls_loss = weighted_loss(logits, y, sample_weights[index], cfg['gce_q'], coefficient, perm if mixed else None, cfg.get('gce_ce_floor', 0.0))
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
                    contrast, contrast_stats = global_trusted_contrast(features, y, logits.detach().argmax(-1) == y, cfg['supcon_temperature'])
                else:
                    sim = features @ features.T / cfg['supcon_temperature']
                    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
                    trusted = logits.detach().argmax(-1) == y
                    positive = (y[:, None] == y[None]) & ~eye & trusted[:, None] & trusted[None]
                    count = positive.sum(-1)
                    logp = sim.masked_fill(eye, -10000.0).log_softmax(-1)
                    if (count > 0).any():
                        contrast = (-(logp * positive).sum(-1) / count.clamp_min(1))[count > 0].mean()
            prior = features.new_zeros(())
            if not mixed and cfg.get('intrinsic_prior_weight', 0):
                prior = model.intrinsic_prior_loss(features, target)
            own_kl, own_drift = (features.new_zeros(()), features.new_zeros(()))
            if own_features is not None:
                own_target = F.normalize(coefficient * own_features[index] + (1 - coefficient) * own_features[index[perm]], dim=-1) if mixed else own_features[index]
                own_drift = (features - own_target).square().sum(-1).mean()
                t = cfg['own_teacher_temperature']
                probability = own_probability[index].clamp_min(1e-09).pow(1 / t)
                probability /= probability.sum(-1, keepdim=True)
                if mixed:
                    permuted = own_probability[index[perm]].clamp_min(1e-09).pow(1 / t)
                    permuted /= permuted.sum(-1, keepdim=True)
                    probability = coefficient * probability + (1 - coefficient) * permuted
                own_kl = F.kl_div((logits / t).log_softmax(-1), probability, reduction='batchmean') * t * t
            geometry = model.projection_geometry_loss() if cfg.get('projection_geometry_weight', 0) else features.new_zeros(())
            loss = cls_loss + cfg['drift'] * drift + cfg['l2sp'] * anchor + cfg['supcon'] * contrast + cfg.get('intrinsic_prior_weight', 0) * prior + cfg.get('projection_geometry_weight', 0) * geometry + cfg.get('own_drift_weight', 0) * own_drift + cfg.get('own_kl_weight', 0) * own_kl
            assert torch.isfinite(loss), f'Nonfinite training loss at step {step}'
            (loss / accumulation).backward()
            if (batch_no + 1) % accumulation:
                continue
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            assert torch.isfinite(norm), f'Nonfinite gradient at step {step}'
            optimizer.step()
            step += 1
            rolling.append([float(loss.detach()), float(cls_loss.detach()), float(drift.detach()), float(anchor.detach()), float(norm)])
            if rank == 0 and (step % 25 == 0 or step == args.until):
                average = np.mean(rolling[-25:], axis=0).tolist()
                state = dict(stage='training', step=step, target=args.until, epoch=epoch, mean_loss=average[0], classification=average[1], drift=average[2], anchor=average[3], gradient_norm=average[4], prior_loss=float(prior.detach()), cutmix_active=cutmixed, mixed=mixed, mixing_coefficient=coefficient, own_feature_drift=float(own_drift.detach()), own_posterior_kl=float(own_kl.detach()), geometry_loss=float(geometry.detach()), seconds=time.monotonic() - started, seconds_per_step=(time.monotonic() - started) / max(1, step - start), gpu=visible, gradient_accumulation=accumulation, effective_batch=cfg['batch']*world*accumulation)
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
