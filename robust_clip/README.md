# Robust CLIP training and inference

CLIP ViT-B/32 fine-tuning with continuous sample reliability weights, Mixup/CutMix,
spatial-region consistency and a single EMA export. The inference recipe uses
full images and centered crops, each with horizontal reflection.

## Runtime

Linux, Python 3.12 and a CUDA GPU. Install `requirements.txt` and place the official
OpenAI `openai/clip-vit-base-patch32` Hugging Face files under
`weights/clip-vit-base-patch32/`. No checkpoint or dataset is distributed here.

## Prepared inputs

The training entry accepts the same prepared artifacts as the deployed pipeline:

- `engine/train_labels.npy`: integer labels in image-cache order, classes 0–499.
- `engine/train_teacher.npy`: frozen official CLIP feature vectors in the same order.
- `image_cache/semi_train_512.npy`: RGB uint8 centered image cache.
- `image_cache/semi_train_complete_512.npy`: corresponding padded-image cache.

These are derived inputs, not additional training data. Derive them only from
the current official training set, preserving exact row alignment. All training
rows participate; no test images are used for fitting or reliability estimation.
The original image augmentation and cache utilities are in `src/data/`.

Run all phases from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --workspace workspace
```

The entry starts with official initialization, preserves optimizer moments for
the second phase, starts a fresh optimizer for robust refinement, supports
checkpoint resume and exports `workspace/model.pt` with EMA parameters. CUDA
attention, BF16, fused AdamW and asynchronous transfers are enabled.

## Inference

Provide an RGB uint8 image cache and a JSON list of filenames in matching order
(a `names` object field is also supported):

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --workspace workspace --checkpoint workspace/model.pt \
  --cache image_cache/semi_test_512.npy --names data/test_names.json \
  --output predictions
```

The fixed views use weights 0.30/0.30 for full images and 0.20/0.20 for centered
0.86 crops; each pair comprises original and reflected input. Temperature is
2.85 with the original probability floor applied before temperature scaling.
Only one model participates. Output is a headerless `pred_results.csv` with
four-digit labels and `pred_results.zip` containing that CSV at its root.

This directory contains source and configuration only. Raw datasets, pretrained
and trained weights, cached features, predictions, run logs and private connection
details must be supplied locally and are excluded from version control.
