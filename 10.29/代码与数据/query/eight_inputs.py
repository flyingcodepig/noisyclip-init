"""Fixed eight-input capture and scatter for the semifinal model."""
from __future__ import annotations
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from query.pixels import Photos, sha, save

SHARED_SOURCE_SHA = 'fef32f6b214347543a626d3ddc54237a4892faa4ea8a21710b988215df7f0d1c'
ARRAYS = {'category_logits.npy': (500, np.float32), 'global768.npy': (768, np.float16),
    'average768.npy': (768, np.float16), 'global512.npy': (512, np.float16),
    'average512.npy': (512, np.float16)}


def size_policy(resolution):
    if resolution not in (448, 640):
        raise ValueError('Only the predeclared448/640 input resolutions are supported')
    return dict(resolution=resolution, whole_resolution=round((resolution * 512 / 448) / 32) * 32,
        source_canvas=512, source_crop=384, views=8,
        whole_size_rule='round((R*512/448)/32)*32',
        source_regions='whole,whole_mirror,center384,four_corner384,complete512',
        resize='PIL BICUBIC short-edge512 center512; tensor bicubic antialias=True clamp0to1')


def make_views(raw, resolution):
    """Return seven R-sized views and one scaled complete-canvas view, on raw.device."""
    policy = size_policy(resolution)
    if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[1:] != (3, 512, 512):
        raise ValueError('Expected the same RGB uint8 center512 source canvas as the shared helper')
    canvas = raw.float() / 255
    resize = lambda pixels, side: F.interpolate(pixels, (side, side), mode='bicubic',
        align_corners=False, antialias=True).clamp(0, 1)
    whole = resize(canvas, resolution)
    center = resize(canvas[:, :, 64:448, 64:448], resolution)
    frames = [whole, whole.flip(-1), center]
    frames.extend(resize(canvas[:, :, y:y + 384, x:x + 384], resolution)
        for y, x in ((0, 0), (0, 128), (128, 0), (128, 128)))
    complete = canvas if policy['whole_resolution'] == 512 else resize(canvas, policy['whole_resolution'])
    return frames, complete


@torch.inference_mode()
def capture(forward, paths, output, entry, model_sha256, part, parts, resolution, binding, batch_size=16):
    if (len(paths) != 49857 or len({path.name for path in paths}) != 49857
            or parts != 4 or part not in range(parts) or batch_size <= 0):
        raise ValueError('Require the complete49857 TEST and one of four strided partitions')
    if sha(ROOT / 'single_encoder_bicubic_queries.py') != SHARED_SOURCE_SHA:
        raise ValueError('The declared shared eight-input preprocessing source changed')
    policy = size_policy(resolution)
    canonical_indices = np.arange(part, len(paths), parts, dtype=np.int64)
    selected = paths[part::parts]
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'query_names.json', [path.name for path in selected])
    arrays = {name: np.lib.format.open_memmap(output / (name + '.writing'), mode='w+',
        dtype=dtype, shape=(len(selected), width)) for name, (width, dtype) in ARRAYS.items()}
    mean = torch.tensor((.48145466, .4578275, .40821073), device='cuda')[None, :, None, None]
    std = torch.tensor((.26862954, .26130258, .27577711), device='cuda')[None, :, None, None]
    cursor, started = 0, time.monotonic()
    with (output / 'raw_predictions.writing.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        for raw, local_index in DataLoader(Photos(selected), batch_size=batch_size,
                num_workers=4, pin_memory=True, shuffle=False, drop_last=False):
            stop = cursor + len(raw)
            if local_index.tolist() != list(range(cursor, stop)):
                raise ValueError('The strided query data loader changed its local image order')
            frames, complete = make_views(raw.cuda(non_blocking=True), resolution)
            packed = torch.stack(frames, dim=1).flatten(0, 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, projected, native = forward((packed - mean) / std)
                complete_logits, complete_projected, complete_native = forward((complete - mean) / std)
            logits = logits.float().reshape(len(raw), 7, 500)
            projected = F.normalize(projected.float(), dim=1).reshape(len(raw), 7, 512)
            native = native.float().reshape(len(raw), 7, 768)
            probabilities = [(logits[:, :2].mean(1) / 2.8).softmax(-1)[:, None],
                (logits[:, 3:] / 2.8).softmax(-1), (complete_logits.float() / 2.8).softmax(-1)[:, None]]
            base = torch.cat(probabilities, dim=1)
            multiplicity = torch.tensor([2.] + [1.] * (base.shape[1] - 1), device='cuda')
            weight = base.max(-1).values * multiplicity[None]
            fused = (base * weight[:, :, None]).sum(1) / weight.sum(1, keepdim=True)
            fused = .875 * fused + .125 * (logits[:, 2] / 2.8).softmax(-1)
            fields = {'category_logits.npy': fused.clamp_min(1e-30).log() * 2.8,
                'global768.npy': F.normalize(complete_native.float(), dim=1),
                'average768.npy': F.normalize(native[:, :2].sum(1), dim=1),
                'global512.npy': F.normalize(complete_projected.float(), dim=1),
                'average512.npy': F.normalize(projected[:, :2].sum(1), dim=1)}
            for name, value in fields.items():
                arrays[name][cursor:stop] = value.cpu().numpy()
            writer.writerows((selected[index].name, f'{label:04d}')
                for index, label in zip(local_index.tolist(), fused.argmax(1).tolist()))
            cursor = stop
            if cursor % 4096 < len(raw):
                print(json.dumps(dict(task='scaled_single_encoder_query', part=part,
                    images=cursor, resolution=resolution, whole_resolution=policy['whole_resolution'],
                    unique_inputs=8, seconds=time.monotonic() - started)), flush=True)
    if cursor != len(selected):
        raise ValueError('The full strided same-model query partition is incomplete')
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError('A scaled same-model query array is nonfinite')
        value.flush()
        (output / (name + '.writing')).replace(output / name)
    (output / 'raw_predictions.writing.csv').replace(output / 'raw_predictions.csv')
    save(output / 'collection_receipt.json', dict(status='complete', entry=entry,
        model_sha256=model_sha256, part=part, parts=parts, partition_policy='canonical_test_files[part::parts]',
        canonical_indices_sha256=hashlib.sha256(canonical_indices.tobytes()).hexdigest(),
        images=cursor, total_TEST_images=49857, unique_inputs=8, single_encoder=True,
        batch_size=batch_size, autocast='bfloat16', physical_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        pid=os.getpid(),
        parameters_updated=False, labels_read=False, source_sha256=sha(Path(__file__)),
        shared_query_source_sha256=SHARED_SOURCE_SHA, size_policy=policy, caller_binding=binding,
        prediction_sha256=sha(output / 'raw_predictions.csv'), query_names_sha256=sha(output / 'query_names.json'),
        files={name: sha(output / name) for name in arrays}, seconds=time.monotonic() - started))


def merge_strided(folders, output, ordered_names, entry, model_sha256, binding, resolution):
    """Scatter one model's four immutable partitions into canonical order; never average."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Strided merging requires an explicit CPU-only environment')
    if (len(folders) != 4 or len(ordered_names) != 49857 or len(set(ordered_names)) != 49857):
        raise ValueError('Require four parts and the complete canonical TEST name list')
    records, coverage = [], np.zeros(49857, dtype=np.uint8)
    raw_labels = [None] * 49857
    policy = size_policy(resolution)
    for part, folder in enumerate(folders):
        receipt = json.loads((folder / 'collection_receipt.json').read_text())
        indices = np.arange(part, 49857, 4, dtype=np.int64)
        names = json.loads((folder / 'query_names.json').read_text())
        if (receipt['status'] != 'complete' or receipt['entry'] != entry or receipt['model_sha256'] != model_sha256
                or receipt['part'] != part or receipt['parts'] != 4 or receipt['images'] != len(indices)
                or receipt['total_TEST_images'] != 49857 or receipt['unique_inputs'] != 8
                or receipt['partition_policy'] != 'canonical_test_files[part::parts]'
                or receipt['canonical_indices_sha256'] != hashlib.sha256(indices.tobytes()).hexdigest()
                or not receipt['single_encoder'] or receipt['parameters_updated'] or receipt['labels_read']
                or receipt['source_sha256'] != sha(Path(__file__))
                or receipt['shared_query_source_sha256'] != SHARED_SOURCE_SHA
                or receipt['size_policy'] != policy or receipt['caller_binding'] != binding
                or names != ordered_names[part::4] or coverage[indices].any()
                or sha(folder / 'query_names.json') != receipt['query_names_sha256']
                or sha(folder / 'raw_predictions.csv') != receipt['prediction_sha256']
                or set(receipt['files']) != set(ARRAYS)):
            raise ValueError('A strided part differs in model, source, eight views, size policy or canonical names')
        for name, (width, dtype) in ARRAYS.items():
            if sha(folder / name) != receipt['files'][name]:
                raise ValueError('A frozen scaled query array changed: ' + name)
            values = np.load(folder / name, mmap_mode='r', allow_pickle=False)
            if values.shape != (len(indices), width) or values.dtype != dtype or not np.isfinite(values).all():
                raise ValueError('A strided scaled query array has an invalid shape, dtype or value')
        with (folder / 'raw_predictions.csv').open(newline='') as stream:
            rows = list(csv.reader(stream))
        if len(rows) != len(indices):
            raise ValueError('A strided raw CSV has missing or extra rows')
        for index, name, row in zip(indices, names, rows):
            if len(row) != 2 or row[0] != name or len(row[1]) != 4 or not row[1].isdigit() or int(row[1]) >= 500:
                raise ValueError('A strided raw prediction has an invalid name or500-class label')
            raw_labels[int(index)] = row[1]
        coverage[indices] += 1
        records.append(receipt)
    if not np.all(coverage == 1) or any(value is None for value in raw_labels):
        raise ValueError('The four parts do not cover every canonical TEST image exactly once')
    output.mkdir(parents=True, exist_ok=False)
    for name, (width, dtype) in ARRAYS.items():
        combined = np.lib.format.open_memmap(output / (name + '.writing'), mode='w+', dtype=dtype, shape=(49857, width))
        for part, folder in enumerate(folders):
            combined[part::4] = np.load(folder / name, mmap_mode='r', allow_pickle=False)
        combined.flush()
        (output / (name + '.writing')).replace(output / name)
    save(output / 'query_names.json', ordered_names)
    with (output / 'raw_predictions.csv').open('x', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(zip(ordered_names, raw_labels))
    save(output / 'collection_receipt.json', dict(status='complete', entry=entry,
        model_sha256=model_sha256, images=49857, parts=4, unique_inputs=8, single_encoder=True,
        parameters_updated=False, labels_read=False, source_sha256=sha(Path(__file__)),
        shared_query_source_sha256=SHARED_SOURCE_SHA, size_policy=policy, caller_binding=binding,
        part_receipt_sha256=[sha(folder / 'collection_receipt.json') for folder in folders],
        partition_policy='four canonical_test_files[part::4], scattered into canonical order',
        unique_TEST_images=49857, missing_TEST_images=0, repeated_TEST_images=0,
        model_or_probability_averaging=False, prediction_sha256=sha(output / 'raw_predictions.csv'),
        query_names_sha256=sha(output / 'query_names.json'), files={name: sha(output / name) for name in ARRAYS}))
