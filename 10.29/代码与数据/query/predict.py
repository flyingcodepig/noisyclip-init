"""Reproduce all TEST predictions from the supplied checkpoint and original images."""
from pathlib import Path as _PackagePath
import sys as _package_sys
_package_sys.path.insert(0,str(_PackagePath(__file__).resolve().parents[1]))
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from ops.runtime import ENTRY, file_hash, save, load_model, gpu_guard
from query.eight_inputs import capture, merge_strided, size_policy
from query.propagation import infer

def paths_for(folder):
    paths = sorted(Path(folder).glob('*.jpg'))
    names = json.loads((ROOT / 'inputs/test_names.json').read_text())
    if [p.name for p in paths] != names or len(names) != 49857 or len(set(names)) != 49857:
        raise ValueError('TEST must contain the complete recorded 49,857 JPG filenames')
    return paths

def binding():
    return dict(entry=ENTRY, route=0., actual_epochs=14, model_count=1,
        source_sha256={p.name: file_hash(p) for p in [ROOT / 'query/predict.py', ROOT / 'ops/runtime.py',
            ROOT / 'scene/foreground_encoder.py', ROOT / 'scene/token_partition.py',
            ROOT / 'scene/evidence_routed_scene.py', ROOT / 'query/propagation.py']},
        size_policy=size_policy(640), graph_geometry='normalized complete and whole-pair native768')

def collect(args):
    with gpu_guard():
        model = load_model(args.checkpoint).cuda()
        def forward(images):
            result = model(images, background_pairing=None)
            return result['scene_logits'], result['object_descriptor'], result['scene_native']
        capture(forward, paths_for(args.test_dir), args.output / 'parts' / f'part{args.part}',
            ENTRY, file_hash(args.checkpoint), args.part, 4, 640, binding(), batch_size=args.batch_size)

def merge(args):
    merge_strided([args.output / 'parts' / f'part{i}' for i in range(4)], args.output / 'bank',
        [p.name for p in paths_for(args.test_dir)], ENTRY, file_hash(args.checkpoint), binding(), 640)

def readout(bank, output, cache=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Readout requires CUDA_VISIBLE_DEVICES=')
    bank, output = Path(bank), Path(output)
    record = json.loads((bank / 'collection_receipt.json').read_text())
    names = json.loads((bank / 'query_names.json').read_text())
    if record['images'] != 49857 or record['entry'] != ENTRY or len(set(names)) != 49857:
        raise ValueError('Unexpected complete prediction bank')
    expected_names = json.loads((ROOT / 'inputs/test_names.json').read_text())
    if names != expected_names:
        raise ValueError('Prediction bank TEST order differs')
    for name, expected in record['files'].items():
        if file_hash(bank / name) != expected:
            raise ValueError('Prediction array changed: ' + name)
    score = torch.from_numpy(np.load(bank / 'category_logits.npy', allow_pickle=False).astype(np.float32))
    geometry = torch.cat([F.normalize(torch.from_numpy(np.load(bank / name, allow_pickle=False).astype(np.float32)), dim=1)
        for name in ['global768.npy', 'average768.npy']], dim=1) / 2**.5
    if score.shape != (49857, 500) or geometry.shape != (49857, 1536):
        raise ValueError('Unexpected array shapes')
    posterior, assigned, info = infer(score, geometry, cache or output / 'neighbors',
        record['model_sha256'] + ':' + file_hash(bank / 'query_names.json'))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'pred_results.csv'
    with target.open('x', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(
            (name, f'{label:04d}') for name, label in zip(names, posterior.argmax(1).tolist()))
    archive = output / 'pred_results.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(target, 'pred_results.csv')
    expected = json.loads((ROOT / 'checks/official_result.json').read_text())['prediction_sha256']
    result = dict(images=len(names), prediction_sha256=file_hash(target),
        submitted_prediction_sha256=expected, matches_submitted=file_hash(target) == expected,
        graph=info, decision='posterior.argmax; assigned result is not used')
    save(output / 'prediction_receipt.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, default=ROOT / 'data/semifinal/test')
    parser.add_argument('--checkpoint', type=Path, default=ROOT.parent / (ENTRY + '-模型文件.pt'))
    parser.add_argument('--output', type=Path, default=ROOT / 'var/runs/predict')
    parser.add_argument('--stage', choices=['all', 'collect', 'merge', 'readout'], default='all')
    parser.add_argument('--part', type=int, choices=range(4))
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8 if args.stage == 'readout' else 4)
    if args.stage == 'all':
        if args.output.exists():
            raise FileExistsError('Choose an empty output directory; use stages explicitly to continue completed parts')
        for part in range(4):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'collect', '--part', str(part),
                '--test-dir', str(args.test_dir.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--output', str(args.output.resolve()), '--batch-size', str(args.batch_size)], check=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        for stage in ['merge', 'readout']:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                '--test-dir', str(args.test_dir.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--output', str(args.output.resolve())], env=env, check=True)
    elif args.stage == 'collect':
        if args.part is None:
            parser.error('--stage collect requires --part')
        collect(args)
    elif args.stage == 'merge':
        merge(args)
    else:
        readout(args.output / 'bank', args.output)

if __name__ == '__main__':
    main()
