"""Build CLIP-refurbished soft labels from the frozen CLIP teacher and prototypes.

TRAIN-only and fully automatic: uses the frozen CLIP teacher features
(train_teacher.npy), the initial class centers (initial_centers.npy), the noisy
given labels (train_labels.npy) and the frozen leave-one-out prototype agreement
(frozen_loo_agreement.npy). No TEST data, no manual cleaning.

The soft label is a convex combination of the one-hot given label and the
temperature-scaled CLIP soft assignment. Samples whose leave-one-out prototype
agrees with the given label keep a small mixing weight (alpha_agree); samples
whose prototype disagrees (likely noisy) mix in more of the CLIP distribution
(alpha_disagree).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
TAU = 0.07
ALPHA_AGREE = 0.1
ALPHA_DISAGREE = 0.5


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=TAU)
    parser.add_argument('--alpha-agree', type=float, default=ALPHA_AGREE)
    parser.add_argument('--alpha-disagree', type=float, default=ALPHA_DISAGREE)
    parser.add_argument('--weight', type=float, default=0.5, help='soft_label_weight written into the generated config')
    args = parser.parse_args()

    teacher = np.load(ENGINE / 'train_teacher.npy')            # (N, 512) normalized
    centers = np.load(ENGINE / 'initial_centers.npy')          # (C, 512) normalized
    labels = np.load(ENGINE / 'train_labels.npy')              # (N,)
    agreement = np.load(ENGINE / 'frozen_loo_agreement.npy').astype(bool)  # (N,)
    num_classes = centers.shape[0]
    assert teacher.shape == (len(labels), 512)

    logits = (teacher @ centers.T) / args.tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    clip_prob = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)  # (N, C)

    soft = np.zeros((len(labels), num_classes), dtype=np.float32)
    soft[np.arange(len(labels)), labels] = 1.0
    alpha = np.where(agreement, args.alpha_agree, args.alpha_disagree)[:, None].astype(np.float32)
    soft = (1.0 - alpha) * soft + alpha * clip_prob

    out = ROOT / 'cache' / 'refurbished_soft_labels.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, soft.astype(np.float32))

    manifest = dict(
        rows=int(len(labels)), classes=int(num_classes),
        tau=args.tau, alpha_agree=args.alpha_agree, alpha_disagree=args.alpha_disagree,
        soft_label_weight=args.weight, test_used=False,
        teacher_sha256=sha(ENGINE / 'train_teacher.npy'),
        centers_sha256=sha(ENGINE / 'initial_centers.npy'),
        labels_sha256=sha(ENGINE / 'train_labels.npy'),
        agreement_sha256=sha(ENGINE / 'frozen_loo_agreement.npy'),
        soft_labels_sha256=sha(out),
    )
    (out.with_suffix('.json')).write_text(json.dumps(manifest, indent=2) + '\n')

    # Emit a ready-to-run config derived from the current stage1.json.
    cfg = json.loads((ROOT / 'stage1.json').read_text())
    cfg['soft_labels'] = 'cache/refurbished_soft_labels.npy'
    cfg['soft_labels_sha256'] = manifest['soft_labels_sha256']
    cfg['soft_label_weight'] = args.weight
    config_path = ROOT / 'stage1_softlabel.json'
    config_path.write_text(json.dumps(cfg, indent=2) + '\n')

    print(json.dumps(dict(
        rows=int(len(labels)), classes=int(num_classes),
        soft_labels=str(out), config=str(config_path),
        disagreement_ratio=float((~agreement).mean()),
        soft_labels_sha256=manifest['soft_labels_sha256'],
    )))


if __name__ == '__main__':
    main()
