"""Train one scene classifier with matched background-residual perturbations.

The partition is a CLS-similarity heuristic. Exchanging the final background
residual does not remove background information already present in CLS or FG.
The two strengths change only the extra decision loss; both execute the same
encoder, perturbation, data order, augmentations and optimizer schedule.
"""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from scene.foreground_encoder import SceneFactorizer

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT
COMMON = Path(os.environ.get('AIC_CACHE_DIR', PROJECT / 'var/cache/train')).resolve()
OFFICIAL = Path(os.environ.get('AIC_CLIP_DIR', PROJECT / 'settings/clip')).resolve()
FORMAT = 'aic-scheduled-scene500-v2'
ENTRY = 'AIC-2026-12985846'
OFFICIAL_SHA = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
PARTITION_SHA = '4482eb2860b33104fc90552c408d0f67ea90a43d68059fe8c29bac8542239a57'
FEATURE_SHA = 'c666170e2ce7ed5fed3c2fe98260ac4820efaf2a45bbca15d2a4f5f963425ca3'
MANIFEST_SHA = 'f3dcebb9bb6dea5da8d1ea644020650edcd798e5cba01ee2bcab93e2df2baabf'
ORIGINAL_MODEL_SHA = 'd325fc07d45f6b4401c971c00312d96de08db6d065fd729c4c5f2f1e910a7dc9'
ORIGINAL_PARTITIONER_SHA = '72d068b78b9ded8a6218aa6ab5e9313cbf22fdf59eec96255996d60ef34ee45a'
from ops.runtime import gpu_guard, physical_device

SOURCE_NAMES = ('training/schedule.py', 'scene/foreground_encoder.py',
                'scene/token_partition.py')
SEED, IMAGE_COUNT, BATCH_SIZE = 12985846, 90197, 88
EPOCH_STEPS = math.ceil(IMAGE_COUNT / BATCH_SIZE)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def atomic_tensor(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    torch.save(value, temporary)
    temporary.replace(path)


def state_hash(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        array = values[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def recipe(exchange_strength):
    if exchange_strength not in (0., 1.):
        raise ValueError('This matched round declares only exchange strengths 0 and 1')
    return dict(seed=SEED, resolution=640, batch_size=BATCH_SIZE, classes=500,
        exchange_strength=float(exchange_strength), epochs=10, first_phase_epochs=8,
        first_phase_objective='CE for hard and mixed observed TRAIN targets',
        last_phase_objective='unmixed observed-label GCE', gce_q=.7,
        first_phase_visual_lr=1.9e-5, first_phase_head_lr=9e-4,
        first_phase_scene_lr=1.5e-4, last_phase_visual_lr=4e-6,
        last_phase_head_lr=8e-5, last_phase_scene_lr=3e-5,
        weight_decay=.045, L2SP=.01, exchange_classification_coefficient=.25,
        exchange_JS_coefficient=.2, gradient_clip=1.,
        first_phase_scope='all vision except fixed official projection',
        last_phase_scope='last4 vision blocks and postLN plus same single head and scene scalars',
        first_phase_batch_pattern='three mixed batches then one unmixed; MixUp/CutMix equiprobable',
        mixup_alpha=.2, cutmix_alpha=1.,
        whole_canvas_probability={'CE': .4, 'GCE': .7}, crop_side_range=[384, 512],
        resize='tensor Bicubic align_corners=False antialias=True; clamp[0,1]',
        horizontal_flip_probability=.5, color_jitter=[.16, .16, .11, .03],
        checkpointing='nonreentrant use_reentrant=False',
        operation='same CLS and FG; perturb only the final background residual',
        all_TRAIN_images_once_per_epoch=True, independent_validation=False,
        teacher_or_peer_fitted_artifacts=False, final_models=1)


def training_inputs():
    from ops.runtime import check_training_inputs
    return check_training_inputs(COMMON, OFFICIAL)


class CanvasRows(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.pixels = np.load(COMMON / 'pixel512_train.npy', mmap_mode='r', allow_pickle=False)
        if self.pixels.shape != (IMAGE_COUNT, 512, 512, 3) or self.pixels.dtype != np.uint8:
            raise ValueError('Unexpected TRAIN pixel shape or dtype')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, row):
        canvas = torch.from_numpy(np.array(self.pixels[row], copy=True)).permute(2, 0, 1)
        return canvas, int(self.labels[row]), int(row)


class SceneDraws:
    """Explicit random draws make paired recipes' image and mixing traces auditable."""
    def __init__(self, device):
        self.device = device
        self.random = np.random.default_rng(SEED + 411)
        self.flip_random = torch.Generator(device=device).manual_seed(SEED + 412)
        self.mean = torch.tensor((.48145466, .4578275, .40821073), device=device)[None, :, None, None]
        self.std = torch.tensor((.26862954, .26130258, .27577711), device=device)[None, :, None, None]
        self.trace = hashlib.sha256()

    @torch.no_grad()
    def image(self, raw, phase):
        crop = [0, 0, 512]
        if self.random.random() < (.6 if phase == 0 else .3):
            side = int(self.random.integers(384, 513))
            crop = [int(self.random.integers(0, 513 - side)),
                    int(self.random.integers(0, 513 - side)), side]
        top, left, side = crop
        pixels = raw.to(self.device, non_blocking=True).float().div_(255.)
        pixels = F.interpolate(pixels[:, :, top:top + side, left:left + side],
            (640, 640), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
        flip = torch.rand(len(raw), generator=self.flip_random, device=self.device) < .5
        pixels = torch.where(flip[:, None, None, None], pixels.flip(-1), pixels)
        factors = [float(self.random.uniform(.84, 1.16)),
                   float(self.random.uniform(.84, 1.16)),
                   float(self.random.uniform(.89, 1.11)),
                   float(self.random.uniform(-.03, .03))]
        ordering = self.random.permutation(4).tolist()
        operations = (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue)
        for choice in ordering:
            pixels = operations[choice](pixels, factors[choice])
        self.trace.update(json.dumps([crop, factors, ordering], separators=(',', ':')).encode())
        self.trace.update(flip.cpu().numpy().tobytes())
        return (pixels - self.mean) / self.std

    @torch.no_grad()
    def target(self, image, labels, phase, step):
        targets = F.one_hot(labels, 500).float()
        mixing = 'none'
        details = []
        if phase == 0 and step % 4 != 3:
            permutation = torch.as_tensor(self.random.permutation(len(labels)), device=self.device)
            if self.random.random() < .5:
                mixing = 'mixup'
                coefficient = float(self.random.beta(.2, .2))
                image = image.mul(coefficient).add(image[permutation], alpha=1 - coefficient)
                details = [coefficient]
            else:
                mixing = 'cutmix'
                fraction = math.sqrt(1 - float(self.random.beta(1., 1.)))
                center_y, center_x = self.random.integers(0, 640, size=2)
                height = width = int(640 * fraction)
                y0, y1 = max(0, int(center_y) - height // 2), min(640, int(center_y) + height // 2)
                x0, x1 = max(0, int(center_x) - width // 2), min(640, int(center_x) + width // 2)
                image[:, :, y0:y1, x0:x1] = image[permutation, :, y0:y1, x0:x1]
                coefficient = 1 - (y1 - y0) * (x1 - x0) / (640 * 640)
                details = [coefficient, y0, y1, x0, x1]
            targets = coefficient * targets + (1 - coefficient) * targets[permutation]
            self.trace.update(permutation.cpu().numpy().tobytes())
        # A nonzero cyclic shift guarantees distinct source rows in every batch.
        shift = int(self.random.integers(1, len(labels)))
        partners = torch.arange(len(labels), device=self.device).roll(shift)
        self.trace.update(json.dumps([mixing, details, shift], separators=(',', ':')).encode())
        return image, targets, partners, mixing

    def rng_state(self):
        return {'numpy_draws': self.random.bit_generator.state,
                'cuda_flip_draws': self.flip_random.get_state()}


def decision_losses(original, exchanged, target, phase):
    logp, logq = original.float().log_softmax(-1), exchanged.float().log_softmax(-1)
    if phase == 0:
        own = -(target * logp).sum(-1)
        other = -(target * logq).sum(-1)
    else:
        # exp preserves low-probability derivatives; expm1's output+1 backward
        # rounds them to zero once its forward result becomes exactly -1.
        own = (target * (1 - (.7 * logp).exp())).sum(-1) / .7
        other = (target * (1 - (.7 * logq).exp())).sum(-1) / .7
    logmiddle = torch.logaddexp(logp, logq) - math.log(2.)
    js = .5 * ((logp.exp() * (logp - logmiddle)).sum(-1)
               + (logq.exp() * (logq - logmiddle)).sum(-1))
    return own.mean(), other.mean(), js.mean()


def set_phase(model, phase):
    for name, value in model.trunk.named_parameters():
        active = not name.startswith('visual_projection.')
        if phase:
            active = name.startswith('vision_model.post_layernorm.') or any(
                name.startswith(f'vision_model.encoder.layers.{layer}.') for layer in range(8, 12))
        value.requires_grad_(active)
    model.trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    visual = [value for value in model.trunk.parameters() if value.requires_grad]
    rates = (1.9e-5, 9e-4, 1.5e-4) if phase == 0 else (4e-6, 8e-5, 3e-5)
    optimizer = torch.optim.AdamW([
        {'params': visual, 'lr': rates[0]},
        {'params': [model.label_basis, model.score_scale], 'lr': rates[1]},
        {'params': [model.fg_level, model.bg_level, model.focus_temperature], 'lr': rates[2], 'weight_decay': 0.},
    ], weight_decay=.045)
    steps = (8 if phase == 0 else 2) * EPOCH_STEPS
    warm = max(1, steps // 25)
    def multiplier(step):
        if step < warm:
            return (step + 1) / warm
        return .5 * (1 + math.cos(math.pi * min(1., (step - warm) / (steps - warm))))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def frozen_vision(model):
    return {name: value.detach().cpu().clone() for name, value in model.trunk.named_parameters()
            if not value.requires_grad}


def check_frozen(model, expected):
    current = dict(model.trunk.named_parameters())
    changed = [name for name, value in expected.items() if not torch.equal(current[name].detach().cpu(), value)]
    if changed:
        raise ValueError('A frozen visual parameter changed: ' + ', '.join(changed))


def probe_gradients(base, extra, model, strength, logits):
    names = ('label_basis', 'fg_level', 'bg_level', 'focus_temperature')
    parameters = [getattr(model, name) for name in names]
    first = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=False)
    second = torch.autograd.grad(extra, parameters, retain_graph=True, allow_unused=False)
    if any(not torch.isfinite(value).all() for value in (*first, *second)):
        raise ValueError('Nonfinite scene objective gradients')
    record = {name: {'original_norm': float(a.float().norm()),
                    'extra_unweighted_norm': float(b.float().norm()),
                    'weighted_extra_norm': float(strength * b.float().norm())}
              for name, a, b in zip(names, first, second)}
    record['exchanged_logit_max_change'] = float((logits['scene_logits'].float() - logits['exchanged_logits'].float()).abs().max())
    record['strength'] = strength
    if record['exchanged_logit_max_change'] <= 0 or record['bg_level']['extra_unweighted_norm'] <= 0:
        raise ValueError('The residual exchange or its scene gradient is inactive')
    return record


@contextmanager
def gpu_guard():
    from ops.runtime import gpu_guard as guard
    with guard() as device:
        yield device


def fit(destination, exchange_strength, probe=0):
    settings = recipe(exchange_strength)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('A run directory already exists; do not overwrite or silently resume')
    if file_hash(HERE / SOURCE_NAMES[1]) != ORIGINAL_MODEL_SHA or file_hash(HERE / SOURCE_NAMES[2]) != ORIGINAL_PARTITIONER_SHA:
        raise ValueError('The matched old scene model or partitioner changed')
    sources = {name: file_hash(HERE / name) for name in SOURCE_NAMES}
    labels, inputs = training_inputs()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    model = SceneFactorizer(OFFICIAL).to(device)
    reference = {name: value.detach().clone() for name, value in model.trunk.named_parameters()
                 if not name.startswith('visual_projection.')}
    official_projection = model.trunk.visual_projection.weight.detach().cpu().clone()
    feature_array = np.load(COMMON / 'official_clip224_features.npy', mmap_mode='r', allow_pickle=False)
    if feature_array.shape != (IMAGE_COUNT, 512):
        raise ValueError('Official feature dimensions do not match TRAIN')
    # Aggregate in FP32; these prototypes use only this team's observed labels.
    features = torch.from_numpy(np.array(feature_array, dtype=np.float32)).to(device)
    targets = torch.as_tensor(labels, device=device)
    with torch.no_grad():
        center = features.mean(0)
        for category in range(500):
            model.label_basis[category].copy_(F.normalize(features[targets == category].mean(0) - .08 * center, dim=0))
    del features, targets
    initial_weights_sha = state_hash(model.state_dict())
    prototype_sha = state_hash({'label_basis': model.label_basis})
    data = CanvasRows(labels)
    draws = SceneDraws(device)
    optimizer, schedule = set_phase(model, 0)
    frozen = frozen_vision(model)
    destination.mkdir(parents=True)
    receipt = dict(format=FORMAT, entry=ENTRY, status='probe_running' if probe else 'training',
        settings=settings, sources=sources, inputs=inputs, history=[], checkpoints={},
        initial_weights_sha256=initial_weights_sha, prototype_sha256=prototype_sha,
        initialization_sha256=OFFICIAL_SHA, worker_pid=os.getpid(),
        physical_gpu=physical_device(),
        started_at=datetime.datetime.now().astimezone().isoformat(),
        expected_training_events=IMAGE_COUNT * 10, expected_optimizer_steps=EPOCH_STEPS * 10,
        expected_tail_batch=IMAGE_COUNT % BATCH_SIZE, other_team_weights_used=False,
        TEST_or_reference_training=False, model_count=1,
        background_operation='final background residual perturbation; original CLS and FG retained')
    atomic_json(destination / 'scene_receipt.json', receipt)
    history, total_steps, total_examples = [], 0, 0
    gradient_report = None
    for epoch in range(10):
        phase = int(epoch >= 8)
        if epoch == 8:
            optimizer, schedule = set_phase(model, 1)
            frozen = frozen_vision(model)
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(IMAGE_COUNT)
        loader_rng = torch.Generator().manual_seed(SEED + epoch + 1000)
        loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=order.tolist(),
            num_workers=8, pin_memory=True, drop_last=False, generator=loader_rng,
            prefetch_factor=2, persistent_workers=False)
        started = time.monotonic()
        seen, sums, mix_counts = [], np.zeros(5, dtype=np.float64), {'none': 0, 'mixup': 0, 'cutmix': 0}
        epoch_images, epoch_steps, label_collisions = 0, 0, 0
        for position, (raw, observed, indices) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            image = draws.image(raw, phase)
            observed = observed.to(device, non_blocking=True)
            image, soft, partners, mixing = draws.target(image, observed, phase, position)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(image, background_pairing=partners)
            base, perturbed, agreement = decision_losses(result['scene_logits'], result['exchanged_logits'], soft, phase)
            extra = .25 * perturbed + .2 * agreement
            preservation = sum((value.float() - reference[name].float()).square().sum()
                for name, value in model.trunk.named_parameters() if name in reference) / len(reference)
            objective = base + exchange_strength * extra + .01 * preservation
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite scene training loss')
            if total_steps == 0:
                gradient_report = probe_gradients(base, extra, model, exchange_strength, result)
                atomic_json(destination / 'scene_gradient_probe.json', gradient_report)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if total_steps == 0:
                if model.trunk.visual_projection.weight.grad is not None:
                    raise ValueError('Official projection acquired a gradient')
                if not any(value.grad is not None and value.grad.float().norm() > 0
                           for name, value in model.trunk.named_parameters()
                           if name.startswith('vision_model.encoder.layers.11.')):
                    raise ValueError('No final visual block gradient')
            optimizer.step(); schedule.step()
            count = len(indices)
            total_steps += 1; total_examples += count; epoch_steps += 1; epoch_images += count
            seen.extend(indices.tolist()); mix_counts[mixing] += 1
            label_collisions += int((observed == observed[partners]).sum())
            sums += count * np.array([float(base.detach()), float(perturbed.detach()),
                float(agreement.detach()), float(preservation.detach()), float(objective.detach())])
            if total_steps <= 3 or total_steps % 200 == 0:
                print(json.dumps(dict(epoch=epoch + 1, phase='CE' if not phase else 'GCE',
                    step=total_steps, images=total_examples, loss=float(objective.detach()),
                    classification=float(base.detach()), exchange_classification=float(perturbed.detach()),
                    exchange_JS=float(agreement.detach()), elapsed_epoch_seconds=time.monotonic() - started,
                    peak_GPU_GiB=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
            if probe and total_steps >= probe:
                break
        del loader
        seen_array = np.asarray(seen, dtype=np.int64)
        expected = order[:len(seen_array)]
        if not np.array_equal(seen_array, expected) or np.unique(seen_array).size != len(seen_array):
            raise ValueError('TRAIN order lost, duplicated or permuted a source row')
        complete_epoch = len(seen_array) == IMAGE_COUNT
        if not probe and (not complete_epoch or epoch_steps != EPOCH_STEPS):
            raise ValueError('Formal epoch did not cover exactly the full TRAIN')
        check_frozen(model, frozen)
        if not torch.equal(model.trunk.visual_projection.weight.detach().cpu(), official_projection):
            raise ValueError('The official projection changed')
        torch.cuda.synchronize()
        memory = torch.cuda.memory_stats()
        report = dict(epoch=epoch + 1, complete_epoch=complete_epoch, phase='CE' if not phase else 'GCE',
            images=epoch_images, unique_images=len(np.unique(seen_array)), steps=epoch_steps,
            source_order_sha256=hashlib.sha256(seen_array.tobytes()).hexdigest(),
            augmentation_trajectory_sha256=draws.trace.hexdigest(),
            mean_classification=float(sums[0] / epoch_images),
            mean_exchange_classification=float(sums[1] / epoch_images),
            mean_exchange_JS=float(sums[2] / epoch_images), mean_L2SP=float(sums[3] / epoch_images),
            mean_total_objective=float(sums[4] / epoch_images), batch_mix_counts=mix_counts,
            exchange_self_pairs=0, exchange_same_observed_label_images=label_collisions,
            last_batch_images=count, seconds=time.monotonic() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            allocation_retries=memory.get('num_alloc_retries', 0), OOMs=memory.get('num_ooms', 0),
            official_projection_unchanged=True, phase_frozen_tensor_count=len(frozen),
            phase_frozen_tensors_unchanged=True,
            foreground_gain=float(.12 * model.fg_level.sigmoid().detach()),
            background_gain=float(.02 + .06 * model.bg_level.sigmoid().detach()))
        history.append(report)
        weights = cpu_weights(model)
        payload = dict(format=FORMAT, entry=ENTRY, classes=500, settings=settings,
            selected_epoch=epoch + 1, history=history.copy(), weights=weights,
            initialization_sha256=OFFICIAL_SHA, training_events=total_examples,
            micro_steps=total_steps, optimizer_steps=total_steps, sources=sources, inputs=inputs,
            TEST_or_reference_training=False, other_team_weights_used=False,
            initial_weights_sha256=initial_weights_sha, model_count=1,
            completed_training=bool(not probe and epoch == 9), probe=bool(probe))
        if probe or epoch in (7, 9):
            filename = 'scene_probe.pt' if probe else ('scene_after8.pt' if epoch == 7 else 'scene_final.pt')
            atomic_tensor(destination / filename, payload)
            receipt['checkpoints'][filename] = dict(sha256=file_hash(destination / filename),
                selected_epoch=epoch + 1, training_events=total_examples, optimizer_steps=total_steps)
        if not probe:
            atomic_tensor(destination / 'training_resume.pt', dict(checkpoint=payload,
                optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                RNG=dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch_CPU=torch.get_rng_state(), torch_CUDA=torch.cuda.get_rng_state_all(),
                    draws=draws.rng_state()), next_epoch=epoch + 1,
                exact_epoch_boundary_saved=True, resume_loader_implemented=False))
        receipt.update(history=history.copy(), training_events=total_examples,
            micro_steps=total_steps, optimizer_steps=total_steps,
            weights_state_sha256=state_hash(weights), gradient_probe=gradient_report,
            status='probe_completed' if probe else ('completed' if epoch == 9 else 'training'))
        atomic_json(destination / 'scene_receipt.json', receipt)
        atomic_json(destination / 'scene_trajectory.json', history)
        print(json.dumps(report), flush=True)
        del payload, weights
        if probe:
            break
    if sources != {name: file_hash(HERE / name) for name in SOURCE_NAMES}:
        raise ValueError('A bound training source changed during the run')
    if file_hash(OFFICIAL / 'pytorch_model.bin') != OFFICIAL_SHA:
        raise ValueError('Official initialization file changed during the run')
    receipt['finished_at'] = datetime.datetime.now().astimezone().isoformat()
    receipt['sources_unchanged_at_finish'] = True
    atomic_json(destination / 'scene_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--exchange-strength', type=float, choices=(0., 1.), required=True)
    parser.add_argument('--probe', type=int, default=0)
    args = parser.parse_args()
    if args.probe < 0:
        parser.error('--probe must be nonnegative')
    with gpu_guard():
        fit(args.destination, args.exchange_strength, args.probe)


if __name__ == '__main__':
    main()
