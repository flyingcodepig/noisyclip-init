"""Matched refinement of foreground contribution using regional class evidence.

One visual encoder and one shared classifier are retained. Both branches keep
FG KL against the original ungated scene prediction; only the contribution
route differs. This entry never starts training on import.
"""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ops.runtime import validate_parent
from training.schedule import (HERE, PROJECT, COMMON, OFFICIAL, OFFICIAL_SHA,
    ENTRY, IMAGE_COUNT, BATCH_SIZE, EPOCH_STEPS, CanvasRows, SceneDraws,
    file_hash, atomic_json, atomic_tensor, state_hash, cpu_weights,
    training_inputs, decision_losses, frozen_vision, check_frozen)
from ops.runtime import gpu_guard
from scene.foreground_alignment import confidence_weighted_alignment
from scene.evidence_routed_scene import EvidenceRoutedScene

FORMAT = 'aic-regional-evidence-gain500-v5'
SEED = 12985846
NEW_EPOCHS = 2


def source_record():
    return {str(path.relative_to(PROJECT)): file_hash(path)
            for path in sorted(HERE.rglob('*.py'))}


def refinement_settings(routing_strength, background_exchange_strength):
    if routing_strength not in (0., 1.) or background_exchange_strength != 1.:
        raise ValueError('Use route0/1 and the fixed f755 background objective')
    return dict(seed=SEED, augmentation_seed=SEED,
        seed_policy='same declared SceneDraws seed reset for both new refinements',
        resolution=640, batch_size=BATCH_SIZE, classes=500, new_epochs=NEW_EPOCHS,
        routing_strength=float(routing_strength), upper_foreground_gain=.08,
        routing_rule='c=argmax(pFG);r=stopgrad(clamp(pFG[c]-pBG[c],0,1));g=g0+eta*max(0,.08-g0)*r',
        routing_stop_gradient=True, main_foreground_agreement_required=False,
        alignment_weight=.05, temperature=2.,
        background_exchange_strength=float(background_exchange_strength),
        visual_lr=1e-6, head_lr=2e-5, scene_lr=1e-5,
        weight_decay=.045, L2SP=.01, gce_q=.7, gradient_clip=1.,
        extra_GCE_weight=.25, JS_weight=.2, pixel_mixing=False,
        scope='last4 vision blocks and postLN plus existing one head and scene scalars',
        frozen='official projection; embeddings,preLN,first8 vision blocks',
        nonreentrant_checkpoint=True, encoder_forwards_per_image=1,
        foreground='original top22percent CLS-related soft pool, LayerNorm then existing postLN/P',
        teacher='ungated original scene logits stopped, identical rule in both route arms',
        teacher_stop_gradient=True, auxiliary_head_stop_gradient=True,
        recipient_gate_reused_for_background_exchange=True,
        no_extra_parameters=True, no_extra_classification_head=True,
        augmentation='SceneDraws phaseGCE: .3 crop;Bicubic640 antialias/clamp;flip/color',
        optimizer_reset=True, RNG_reset=True, exact_optimizer_resume=False,
        inference='EvidenceRoutedScene with declared route and no donor, one encoder',
        model_count=1, teacher_or_peer_fitted_artifacts=False,
        independent_validation=False, reference_training=False)


def parent_record(path):
    return validate_parent(path, selected_epoch=12, receipt_name='alignment_receipt.json')


def refinement_optimizer(model):
    for name, parameter in model.trunk.named_parameters():
        allowed = name.startswith('vision_model.post_layernorm.') or any(
            name.startswith(f'vision_model.encoder.layers.{layer}.') for layer in range(8, 12))
        parameter.requires_grad_(allowed)
    model.trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW([
        dict(params=[value for value in model.trunk.parameters() if value.requires_grad], lr=1e-6),
        dict(params=[model.label_basis, model.score_scale], lr=2e-5),
        dict(params=[model.fg_level, model.bg_level, model.focus_temperature], lr=1e-5, weight_decay=0.),
    ], weight_decay=.045)
    total = NEW_EPOCHS * EPOCH_STEPS
    warmup = max(1, total // 25)
    def scale(step):
        return ((step + 1) / warmup if step < warmup else
            .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / (total - warmup)))))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def routing_gradient_probe(result, alignment, model, routing_strength, encoder_calls):
    if encoder_calls != 1:
        raise ValueError('Regional routing requires exactly one encoder forward')
    r = result['routing_reliability']
    if r.requires_grad or result['routing_selected_class'].requires_grad:
        raise ValueError('Routing evidence must be stopped')
    head_gradients = torch.autograd.grad(alignment, (model.label_basis, model.score_scale),
        retain_graph=True, allow_unused=True)
    teacher_gradient = torch.autograd.grad(alignment, result['ungated_scene_logits'],
        retain_graph=True, allow_unused=True)[0]
    if any(value is not None for value in head_gradients) or teacher_gradient is not None:
        raise ValueError('Auxiliary shared head or ungated teacher is not stopped')
    names = ['trunk.vision_model.encoder.layers.8.self_attn.q_proj.weight',
             'trunk.vision_model.encoder.layers.11.self_attn.q_proj.weight',
             'trunk.vision_model.post_layernorm.weight']
    parameters = dict(model.named_parameters())
    auxiliary = torch.autograd.grad(alignment, [parameters[name] for name in names], retain_graph=True)
    norms = {name: float(value.float().norm()) for name, value in zip(names, auxiliary)}
    if any(not torch.isfinite(value).all() for value in auxiliary) or any(value <= 0 for value in norms.values()):
        raise ValueError('FG alignment must still reach visual and postLN parameters')
    gain = result['effective_foreground_gain']
    difference = float((result['scene_logits'] - result['ungated_scene_logits']).float().abs().max().detach())
    if routing_strength == 0.:
        if result['scene_logits'] is not result['ungated_scene_logits'] or difference != 0.:
            raise ValueError('Baseline must reuse the exact original main tensor')
        if not torch.equal(gain, result['original_foreground_gain'].expand_as(gain)):
            raise ValueError('Baseline foreground gain changed')
    elif difference <= 0 or float((gain - result['original_foreground_gain']).max().detach()) <= 0:
        raise ValueError('Dynamic contribution does not affect the actual main path')
    if not bool(((r >= 0) & (r <= 1)).all()) or not bool((gain <= .080001).all()):
        raise ValueError('Routing evidence/gain escaped its declared range')
    return dict(routing_strength=float(routing_strength), encoder_forward_calls=encoder_calls,
        routing_has_gradient=False, teacher_ungated_logit_gradient=None,
        auxiliary_class_weight_gradient=None, auxiliary_scale_gradient=None,
        visual_auxiliary_gradient_norms=norms, original_main_exact=(routing_strength == 0.),
        main_logit_max_abs_change=difference,
        reliability_mean=float(r.mean()), reliability_min=float(r.min()), reliability_max=float(r.max()),
        gain_mean=float(gain.mean().detach()), gain_min=float(gain.min().detach()), gain_max=float(gain.max().detach()),
        main_foreground_agreement_required=False, recipient_gate_reused_for_exchange=True)


def fit(parent_path, destination, routing_strength, probe=0):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('A foreground alignment directory already exists')
    sources = source_record()
    parent, parent_proof, official = parent_record(parent_path)
    background_strength = parent_proof['settings']['background_exchange_strength']
    settings = refinement_settings(routing_strength, background_strength)
    labels, inputs = training_inputs()
    if parent['inputs'] != inputs:
        raise ValueError('The refinement TRAIN identity differs from the actual parent')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    model = EvidenceRoutedScene(official, routing_strength=routing_strength).to(device)
    reference = {name: parameter.detach().clone() for name, parameter in model.trunk.named_parameters()
                 if not name.startswith('visual_projection.')}
    official_projection = model.trunk.visual_projection.weight.detach().cpu().clone()
    model.load_state_dict(parent['weights'], strict=True)
    initial_sha = state_hash(model.state_dict())
    if initial_sha != state_hash(parent['weights']):
        raise ValueError('The refinement did not load the exact parent state')
    optimizer, schedule = refinement_optimizer(model)
    frozen = frozen_vision(model)
    if len(frozen) != 134 or 'visual_projection.weight' not in frozen:
        raise ValueError('Unexpected first8-plus-projection frozen scope')
    selected_changes = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
        if name in {'label_basis', 'fg_level', 'bg_level', 'focus_temperature',
                    'trunk.vision_model.encoder.layers.8.self_attn.q_proj.weight',
                    'trunk.vision_model.encoder.layers.11.self_attn.q_proj.weight'}}
    parent_history = parent['history']
    del parent
    dataset, draws = CanvasRows(labels), SceneDraws(device)
    destination.mkdir(parents=True)
    receipt = dict(format=FORMAT, entry=ENTRY, status='probe_running' if probe else 'training',
        settings=settings, parent=parent_proof, sources=sources, inputs=inputs,
        history=parent_history.copy(), refinement_history=[], checkpoints={},
        initialization_sha256=OFFICIAL_SHA, initial_weights_sha256=initial_sha,
        expected_new_training_events=180394, expected_new_optimizer_steps=2050,
        new_epochs=NEW_EPOCHS, expected_total_epochs=parent_proof['selected_epoch'] + NEW_EPOCHS,
        PID=os.getpid(), GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        started_at=datetime.datetime.now().astimezone().isoformat(),
        TEST_or_reference_training=False, other_team_weights_used=False,
        model_count=1, model_weight_averaging=False)
    atomic_json(destination / 'regional_receipt.json', receipt)
    new_history, new_events, new_steps = [], 0, 0
    gradient_report = None
    for local_epoch in range(NEW_EPOCHS):
        model.train()
        global_epoch = parent_proof['selected_epoch'] + local_epoch + 1
        order = np.random.default_rng(SEED + local_epoch).permutation(IMAGE_COUNT)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=order.tolist(),
            num_workers=8, pin_memory=True, drop_last=False,
            generator=torch.Generator().manual_seed(SEED + local_epoch + 1000),
            prefetch_factor=2, persistent_workers=False)
        start = time.monotonic()
        seen, sums, steps, images, collisions = [], np.zeros(9), 0, 0, 0
        for batch_index, (raw, observed, indices) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            image = draws.image(raw, 1)
            observed = observed.to(device, non_blocking=True)
            image, target, pairing, mixing = draws.target(image, observed, 1, batch_index)
            if mixing != 'none':
                raise ValueError('Foreground refinement must use unmixed observed labels')
            encoder_calls = []
            hook = model.trunk.vision_model.register_forward_hook(
                lambda *_args: encoder_calls.append(1)) if new_steps == 0 else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(image, background_pairing=pairing, return_diagnostics=(new_steps == 0))
            if hook is not None:
                hook.remove()
            base, altered, agreement = decision_losses(result['scene_logits'], result['exchanged_logits'], target, 1)
            alignment, alignment_info = confidence_weighted_alignment(
                result['ungated_scene_logits'], result['foreground_logits'], temperature=2.)
            preservation = sum((parameter.float() - reference[name].float()).square().sum()
                for name, parameter in model.trunk.named_parameters() if name in reference) / len(reference)
            original_objective = base + background_strength * (.25 * altered + .2 * agreement) + .01 * preservation
            loss = original_objective + .05 * alignment
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite foreground alignment objective')
            if new_steps == 0:
                gradient_report = routing_gradient_probe(result, alignment, model, routing_strength, len(encoder_calls))
                atomic_json(destination / 'regional_gradient_probe.json', gradient_report)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if any(parameter.grad is not None for name, parameter in model.trunk.named_parameters() if name in frozen):
                raise ValueError('A declared frozen early/P parameter acquired a gradient')
            optimizer.step(); schedule.step()
            count = len(indices)
            new_steps += 1; new_events += count; steps += 1; images += count
            seen.extend(indices.tolist())
            collisions += int((observed == observed[pairing]).sum())
            sums += count * np.array([float(base.detach()), float(altered.detach()),
                float(agreement.detach()), float(preservation.detach()), float(loss.detach()),
                float(alignment.detach()), float(alignment_info['confidence'].mean()),
                float(result['routing_reliability'].mean()), float(result['effective_foreground_gain'].detach().mean())])
            if new_steps <= 3 or new_steps % 200 == 0:
                print(json.dumps(dict(local_epoch=local_epoch + 1, epoch=global_epoch,
                    step=new_steps, new_images=new_events, loss=float(loss.detach()),
                    GCE=float(base.detach()), altered_GCE=float(altered.detach()),
                    background_JS=float(agreement.detach()), FG_alignment=float(alignment.detach()),
                    elapsed_epoch_seconds=time.monotonic() - start,
                    peak_GPU_GiB=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
            if probe and new_steps >= probe:
                break
        del loader
        observed_order = np.asarray(seen, dtype=np.int64)
        if (not np.array_equal(observed_order, order[:len(observed_order)])
                or np.unique(observed_order).size != len(observed_order)):
            raise ValueError('Foreground training lost or repeated a source row')
        complete = images == IMAGE_COUNT
        if not probe and (not complete or steps != EPOCH_STEPS or count != 85):
            raise ValueError('The formal refinement did not use the declared full TRAIN budget')
        check_frozen(model, frozen)
        if not torch.equal(model.trunk.visual_projection.weight.detach().cpu(), official_projection):
            raise ValueError('The official projection changed')
        deltas = {name: float((dict(model.named_parameters())[name].detach().cpu() - initial).norm())
                  for name, initial in selected_changes.items()}
        if any(deltas[name] <= 0 for name in ['label_basis',
                'trunk.vision_model.encoder.layers.8.self_attn.q_proj.weight',
                'trunk.vision_model.encoder.layers.11.self_attn.q_proj.weight']):
            raise ValueError('The allowed visual scope or single classifier did not update')
        torch.cuda.synchronize()
        memory = torch.cuda.memory_stats()
        report = dict(epoch=global_epoch, refinement_epoch=local_epoch + 1,
            complete_epoch=complete, phase='GCE_regional_evidence', images=images,
            unique_images=len(np.unique(observed_order)), steps=steps, last_batch_images=count,
            source_order_sha256=hashlib.sha256(observed_order.tobytes()).hexdigest(),
            augmentation_trajectory_sha256=draws.trace.hexdigest(),
            mean_classification=float(sums[0] / images), mean_perturbed_classification=float(sums[1] / images),
            mean_JS=float(sums[2] / images), mean_L2SP=float(sums[3] / images),
            mean_objective=float(sums[4] / images), mean_foreground_alignment=float(sums[5] / images),
            mean_stopped_main_confidence=float(sums[6] / images),
            mean_routing_reliability=float(sums[7] / images),
            mean_effective_foreground_gain=float(sums[8] / images), seconds=time.monotonic() - start,
            exchange_self_pairs=0, exchange_same_observed_label_images=collisions,
            pixel_mixed_batches=0, official_projection_unchanged=True,
            phase_frozen_tensor_count=len(frozen), phase_frozen_tensors_unchanged=True,
            frozen_early_parameters_equal_parent=True, actual_parameter_change_norms=deltas,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            allocation_retries=memory.get('num_alloc_retries', 0), OOMs=memory.get('num_ooms', 0))
        new_history.append(report)
        full_history = parent_history + new_history
        weights = cpu_weights(model)
        payload = dict(format=FORMAT, entry=ENTRY, classes=500, settings=settings,
            parent=parent_proof, selected_epoch=global_epoch,
            actual_total_epochs=parent_proof['selected_epoch'] + sum(row['complete_epoch'] for row in new_history),
            new_epochs=sum(row['complete_epoch'] for row in new_history),
            history=full_history, refinement_history=new_history.copy(),
            weights=weights, sources=sources, inputs=inputs, initialization_sha256=OFFICIAL_SHA,
            initial_weights_sha256=initial_sha,
            training_events=parent_proof['training_events'] + new_events,
            micro_steps=parent_proof['micro_steps'] + new_steps,
            optimizer_steps=parent_proof['optimizer_steps'] + new_steps,
            new_training_events=new_events, new_micro_steps=new_steps, new_optimizer_steps=new_steps,
            probe=bool(probe), completed_training=bool(not probe and local_epoch == 1),
            TEST_or_reference_training=False, other_team_weights_used=False,
            model_count=1, model_weight_averaging=False,
            frozen_early_parameters_equal_parent=True, official_projection_unchanged=True)
        if probe or local_epoch == 1:
            filename = 'regional_probe.pt' if probe else 'regional_final.pt'
            atomic_tensor(destination / filename, payload)
            receipt['checkpoints'][filename] = dict(sha256=file_hash(destination / filename),
                selected_epoch=global_epoch, training_events=payload['training_events'],
                optimizer_steps=payload['optimizer_steps'], new_training_events=new_events,
                new_optimizer_steps=new_steps)
        if not probe:
            atomic_tensor(destination / 'regional_resume.pt', dict(checkpoint=payload,
                optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(),
                RNG=dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch_CPU=torch.get_rng_state(), torch_CUDA=torch.cuda.get_rng_state_all(),
                    draws=draws.rng_state()), next_refinement_epoch=local_epoch + 1,
                exact_epoch_boundary_saved=True, resume_loader_implemented=False))
        receipt.update(history=full_history, refinement_history=new_history.copy(),
            actual_total_epochs=payload['actual_total_epochs'], new_epochs=payload['new_epochs'],
            training_events=payload['training_events'], micro_steps=payload['micro_steps'],
            optimizer_steps=payload['optimizer_steps'], new_training_events=new_events,
            new_micro_steps=new_steps, new_optimizer_steps=new_steps,
            weights_state_sha256=state_hash(weights), gradient_probe=gradient_report,
            frozen_early_parameters_equal_parent=True, official_projection_unchanged=True,
            status='probe_completed' if probe else ('completed' if local_epoch == 1 else 'training'))
        atomic_json(destination / 'regional_receipt.json', receipt)
        atomic_json(destination / 'regional_trajectory.json', new_history)
        print(json.dumps(report), flush=True)
        del payload, weights
        if probe:
            break
    if sources != source_record() or file_hash(PROJECT / parent_proof['path']) != parent_proof['sha256']:
        raise ValueError('A bound source or parent changed during refinement')
    if file_hash(PROJECT / parent_proof['receipt_path']) != parent_proof['receipt_sha256']:
        raise ValueError('The completed parent receipt changed during refinement')
    if file_hash(OFFICIAL / 'pytorch_model.bin') != OFFICIAL_SHA:
        raise ValueError('The official initialization file changed')
    receipt['sources_unchanged_at_finish'] = True
    receipt['finished_at'] = datetime.datetime.now().astimezone().isoformat()
    atomic_json(destination / 'regional_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--routing-strength', type=float, choices=(0., 1.), required=True)
    parser.add_argument('--probe', type=int, default=0)
    args = parser.parse_args()
    if args.probe < 0:
        parser.error('--probe must be nonnegative')
    with gpu_guard():
        fit(args.parent, args.destination, args.routing_strength, args.probe)


if __name__ == '__main__':
    main()
