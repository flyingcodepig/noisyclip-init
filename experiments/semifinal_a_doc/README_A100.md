# A100 上运行半决赛 A 路线重实现

本包按 `01_全部实验与算法路线总账.md` 和 `02_A当前完整算法与最初方案对照.md` 重写。参数以两份文档为准。它不是队友的原始源码或 77.6293 分模型，尚未在 A100 上端到端验证；运行后必须把训练日志、模型哈希与官方得分作为新实验记录，不能声称逐位复现队友结果。

## 输入和输出

- 半决赛 `train.zip` 解压到 `data/train/`，直接包含 `0000/` 至 `0499/`，共 90,197 张 JPG。
- 半决赛 `test.zip` 解压到 `data/test/`，直接包含 49,857 张 JPG。
- 官方 OpenAI CLIP ViT-B/32 权重由 `fetch_weights.py` 下载到 `weights/clip-vit-base-patch32/pytorch_model.bin`，并核对 SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`。
- `prepare.py` 只读取 TRAIN；`train.py` 只读取准备好的 TRAIN 清单和派生输入；`deliver.py` 才读取 TEST。
- `runs/a/final_ema.pt` 是本次训练导出的单模型 EMA 权重；`output/submission.zip` 内应只有 `pred_results.csv`。

## JupyterLab Terminal 指令

在终端中执行，按上传位置调整前两行：

```bash
mkdir -p ~/aic_a100
cd ~/aic_a100
unzip -q /path/to/aic_a100_replica.zip
cd replica_a

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# PyTorch 和 torchvision 必须使用租用镜像中相互匹配的 CUDA 构建。
python -m pip install -r requirements.txt
python - <<'PY'
import torch, torchvision, transformers
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__)
print('GPU', torch.cuda.get_device_name(0), 'memory_GiB', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PY

python fetch_weights.py
mkdir -p data/train data/test prepared runs
python -m zipfile -e /path/to/train.zip data/train
python -m zipfile -e /path/to/test.zip data/test
python - <<'PY'
from data import inventory
print('TRAIN', len(inventory('data/train')))
print('TEST', len(inventory('data/test', train=False)))
PY

python prepare.py --train-root data/train --official-weights weights/clip-vit-base-patch32 --output prepared 2>&1 | tee prepare.log

# 独立的 2 步烟雾检查；它不作为正式训练的起点。
python train.py --prepared prepared --official-weights weights/clip-vit-base-patch32 --run runs/smoke --stop-after 2 2>&1 | tee smoke.log

# 正式从官方 CLIP 初始化训练 4862 步。
python train.py --prepared prepared --official-weights weights/clip-vit-base-patch32 --run runs/a 2>&1 | tee train.log

python deliver.py --test-root data/test --official-weights weights/clip-vit-base-patch32 --checkpoint runs/a/final_ema.pt --output output 2>&1 | tee deliver.log
python validate_submission.py --test-root data/test --zip output/submission.zip
sha256sum runs/a/final_ema.pt output/submission.zip
```

若训练中断，保留相同代码、配置和 `runs/a/last.pt`，执行：

```bash
python train.py --prepared prepared --official-weights weights/clip-vit-base-patch32 --run runs/a --resume 2>&1 | tee -a train.log
```

如 A100 40GB 在烟雾检查时显存不足，使用 `--micro-batch 32` 将微批降为 32、累积改为 6，仍保持有效 batch 192。正式训练和续训必须使用同一选项；这一资源适配可能使数值轨迹与文档中的 96×2 不同。

## 运行前限制

本实现没有队友原版的配置/权重对照测试。`prepare.py` 的冻结教师输入、增强与可信队列是按文档重新解释的实现，可能与原代码存在细节差异。先运行烟雾检查并查看 loss 与显存，再决定是否投入完整 4862 步。
