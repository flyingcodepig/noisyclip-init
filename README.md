# NoisyCLIP

NoisyCLIP is an executable F02 implementation of a noisy-label fine-grained
image classification pipeline based on official OpenAI CLIP ViT-B/32 weights.

It includes deterministic data auditing, fixed manifests, B0/B1/B2 model
assembly, the B3-B6 robust-training modules, checkpoint/resume, fixed-validation
evaluation, single-student export, offline inference, and submission validation.
The U1-U6 upper-bound configurations remain intentionally blocked until their
individual integration experiments are implemented; they never silently fall
back to a different method.

## Development

Install editable development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the quality gates:

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
```

## Main workflow

Set the four paths used by `configs/base.yaml`, run the immutable data audit,
then start a uniquely named experiment:

```bash
export NOISYCLIP_TRAIN_ROOT=/data/train
export NOISYCLIP_TEST_ROOT=/data/test
export NOISYCLIP_RUN_ROOT=/runs
export XDG_CACHE_HOME=/cache/xdg

python -m noisyclip.cli.audit_data --config configs/base.yaml
python -m noisyclip.cli.train --config configs/base.yaml --run-id b0-seed-20260806
python -m noisyclip.cli.evaluate --run-dir /runs/b0-seed-20260806
```

Training is offline-only: the official `ViT-B-32.pt` must already exist under
`$XDG_CACHE_HOME`. To resume a failed run, pass both its original run ID and an
explicit committed checkpoint:

```bash
python -m noisyclip.cli.train \
  --config configs/base.yaml \
  --run-id b0-seed-20260806 \
  --resume /runs/b0-seed-20260806/checkpoints/last.pt
```

## 已提交的 71.06 分版本

用户报告线上成绩 **71.0621387283237** 的原始四视图分类源码位于 [`submitted_71_06/`](submitted_71_06/)。训练、推理配置和归档校验值一并保留。

请先阅读 [源码快照与模型恢复说明](submitted_71_06/SOURCE_EXPORT.md)。大体积模型权重、数据索引和原始预测结果未提交到 Git，使用说明中的脚本从本地完整归档恢复后，再按原版 README 复现。
