"""Portable paths, checkpoint checks and exclusive CUDA execution."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT
OFFICIAL_SHA = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
FEATURE_SHA = 'c666170e2ce7ed5fed3c2fe98260ac4820efaf2a45bbca15d2a4f5f963425ca3'
PIXEL_SHA = '943a8e6d79d339e95bd77f96e85ac517f89785b70fd42e748081061781474c7d'
MANIFEST_SHA = 'f3dcebb9bb6dea5da8d1ea644020650edcd798e5cba01ee2bcab93e2df2baabf'
ENTRY = 'AIC-2026-12985846'

def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def state_hash(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        a = values[name].detach().cpu().contiguous().numpy()
        for value in (name.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()):
            digest.update(value)
    return digest.hexdigest()

def physical_device():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if not selected or ',' in selected:
        raise ValueError('Select one CUDA GPU per process with CUDA_VISIBLE_DEVICES')
    return int(subprocess.check_output(['nvidia-smi', '-i', selected,
        '--query-gpu=index', '--format=csv,noheader,nounits'], text=True).strip())

@contextmanager
def gpu_guard():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    physical = physical_device()
    with open(f'/tmp/aic-reference-physical-gpu{physical}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lines = subprocess.check_output(['nvidia-smi', '-i', selected,
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        if any(line.strip().isdigit() and int(line.strip()) != os.getpid() for line in lines):
            raise RuntimeError('The selected GPU is occupied; choose an idle GPU')
        yield physical

def check_training_inputs(cache, official):
    cache, official = Path(cache), Path(official)
    manifest = PROJECT / 'inputs/train_manifest.jsonl'
    features = cache / 'official_clip224_features.npy'
    pixel = cache / 'pixel512_train.npy'
    if file_hash(manifest) != MANIFEST_SHA or file_hash(features) != FEATURE_SHA:
        raise ValueError('The recorded TRAIN order or official frozen features changed')
    if file_hash(official / 'pytorch_model.bin') != OFFICIAL_SHA:
        raise ValueError('The official CLIP initialization changed')
    receipt_path = cache / 'cache_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('status') != 'complete' or receipt.get('pixels_sha256') != PIXEL_SHA
            or receipt.get('features_sha256') != FEATURE_SHA or pixel.stat().st_size != 70933807232):
        raise ValueError('Build and verify the full TRAIN cache with scene.sh cache')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    labels = np.asarray([row['label'] for row in rows], dtype=np.int64)
    if len(rows) != 90197 or set(labels) != set(range(500)) or len({r['relative_path'] for r in rows}) != 90197:
        raise ValueError('Unexpected TRAIN identity')
    inputs = dict(initialization_sha256=OFFICIAL_SHA, manifest_sha256=MANIFEST_SHA,
        features_sha256=FEATURE_SHA, pixels_sha256=PIXEL_SHA, TRAIN_images=90197,
        TRAIN_manifest_and_labels_equal=True, TEST_or_reference_training=False)
    return labels, inputs

def validate_parent(path, selected_epoch, receipt_name):
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    receipt_path = path.with_name(receipt_name)
    receipt = json.loads(receipt_path.read_text())
    digest = file_hash(path)
    record = receipt['checkpoints'][path.name]
    if (payload['entry'] != ENTRY or payload['classes'] != 500
            or payload['selected_epoch'] != selected_epoch or not payload['completed_training']
            or payload.get('probe') or payload['model_count'] != 1
            or record['sha256'] != digest or receipt['status'] != 'completed'
            or not receipt['sources_unchanged_at_finish']
            or payload['initialization_sha256'] != OFFICIAL_SHA
            or payload['training_events'] != selected_epoch * 90197
            or payload['optimizer_steps'] != selected_epoch * 1025):
        raise ValueError('The parent is not a completed checkpoint from the required stage')
    for key in ['settings', 'inputs', 'history', 'sources']:
        if payload[key] != receipt[key]:
            raise ValueError('The parent checkpoint and receipt disagree: ' + key)
    if payload['TEST_or_reference_training'] or payload['other_team_weights_used']:
        raise ValueError('Unexpected parent training inputs')
    for relative, expected in payload['sources'].items():
        candidates = [PROJECT / relative, SOURCE / relative]
        observed = next((p for p in candidates if p.is_file()), None)
        if observed is None or file_hash(observed) != expected:
            raise ValueError('A parent training source changed: ' + relative)
    official = Path(os.environ.get('AIC_CLIP_DIR', PROJECT / 'settings/clip')).resolve()
    initialized = torch.load(official / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    if not torch.equal(payload['weights']['trunk.visual_projection.weight'], initialized['visual_projection.weight']):
        raise ValueError('The official visual projection changed')
    proof = dict(path=os.path.relpath(path, PROJECT), sha256=digest,
        receipt_path=os.path.relpath(receipt_path, PROJECT), receipt_sha256=file_hash(receipt_path),
        selected_epoch=selected_epoch, training_events=payload['training_events'],
        micro_steps=payload['micro_steps'], optimizer_steps=payload['optimizer_steps'],
        format=payload['format'], settings=payload['settings'], original_projection_verified=True)
    return payload, proof, official

def load_model(checkpoint):
    from scene.evidence_routed_scene import EvidenceRoutedScene
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['entry'] != ENTRY or payload['classes'] != 500 or payload['model_count'] != 1:
        raise ValueError('Wrong model identity')
    route = payload.get('routing_strength', payload.get('settings', {}).get('routing_strength'))
    if route != 0. or not payload['completed_training'] or payload.get('probe'):
        raise ValueError('This release uses the completed route-zero scene model')
    if payload.get('tensor_state_sha256') and state_hash(payload['weights']) != payload['tensor_state_sha256']:
        raise ValueError('Model tensor checksum mismatch')
    model = EvidenceRoutedScene(PROJECT / 'settings/clip',
        routing_strength=0., initialize_pretrained=False)
    model.load_state_dict(payload['weights'], strict=True)
    model.trunk.gradient_checkpointing_disable()
    return model.eval().requires_grad_(False)
