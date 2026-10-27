# AIC-2026-12985846 四视图分类版：代码与模型

本包归档用户已提交并报告线上成绩 **71.0621387283237** 的版本。
推理为同一个最终模型的加权四视图分类，不使用图传播或类别数量均衡。
用户报告的是总分，平台实际评测图片清单未公开；本包输出覆盖现有测试包的 49,857 张。

## 文件对应关系

| 位置 | 内容 |
| --- | --- |
| `model/model.pt` | 已提交版本对应的第二阶段模型，step=2048，500 类 |
| `model/stage1_model.pt` | 同一训练流程的第一阶段模型，step=4096，仅供归档 |
| `code/weights/clip-vit-base-patch32/` | 官方初始化权重、模型配置、图像预处理配置 |
| `code/engine/`、`code/src/` | 训练、模型结构、推理和增强的实际实现 |
| `code/semifinal_stage1.json`、`code/semifinal_stage2.json` | 实际使用的单卡训练配置 |
| `code/inference_policy.json` | 已提交版本使用的视图、温度和权重；其中图参数不被四视图复现入口使用 |
| `results/pred_results.csv`、`pred_results.zip` | 当时四视图分类版的原始输出；可直接核对 |
| `metadata/` | 实际环境、输入校验值、模型清单及原推理特征绑定 |
| `bundle_manifest.json`、`SHA256SUMS.txt` | 文件大小与 SHA256 |

当前新一轮 B0/B1 优化训练模型不属于本包。原训练包、测试包及巨大的图片/特征缓存
不重复打入归档；复现时使用项目已有的原始 `train.zip` / `test.zip`。
模型提供的是权重版检查点，不包含原训练优化器状态；从头复现入口会重新训练并生成
含优化器的 `runs/stage1/last.pt`，供第二阶段继续使用。
包内不含 SSH 登录脚本或服务器密码。

## 关键校验值

- 最终模型：`cd30a7d8afb6785d9d889dde65726315cc67b5e56c8ecc1a98ce1da59cc8d3fc`
- 四视图 CSV：`996e481af5b93d982cf483fc7cdf56ce177f2e17c7fa6c40ee874ac6ac7cd73d`
- 原提交 ZIP：`4e0d519328cdd7d94a036d424db6f71830e24e37bd8c74ff55aac7a95556156d`
- 官方初始化权重：`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`

解压后先在本目录执行（只需 Python 标准库）：

```bash
python verify_package.py
```

## 实际环境

原训练和推理在 Linux、单张 RTX 4090 24GB 上完成，Python 3.12，
PyTorch 2.12.1+cu130、torchvision 0.27.1+cu130、transformers 4.57.1、
NumPy 2.4.6、Pillow 12.2.0。以 `metadata/semifinal_actual_environment.json`
和 `metadata/requirements.actual.txt` 为准。CUDA/PyTorch 需使用匹配的 GPU 运行环境。
原推理微批为 48；训练为微批 48、梯度累积 4 次，有效批量 192。
对比损失仍按 48 张微批计算，和原复赛四卡跨 192 张的数值行为有区别。

## 用最终模型重新生成四视图结果

将原始测试包放到 `code/test.zip`。测试缓存需约 78.4GB，加上数据包和输出，
建议有至少 100GB 可用空间。该入口检查原始数据包和模型 SHA256，使用所有原始文件名。

在本包根目录运行：

```bash
python code/reproduce_submitted_four_view.py
```

输出位于 `reproduced_results/`：

- `pred_results.csv`：无表头，两列，文件名与 0000–0499 四位类别编号。
- `pred_results.zip`：根目录只包含 `pred_results.csv`。
- `reproduction_manifest.json`：是否与原始 CSV 的 SHA256 完全一致。

每次复现请使用空的结果目录；例如：

```bash
python code/reproduce_submitted_four_view.py --output reproduced_results_2
```

四视图及权重为完整画面 1.0/不翻转 0.23、中心 1.0/不翻转 0.16、
中心 0.86/翻转 0.21、中心 0.74/不翻转 0.40；先对每视图概率按温度 2.85 变换，
再进行加权平均和 argmax。第一、第二视图在原数据上完全相同，仍保留原运算次序。
本入口没有调用 `readout.py` 的图传播。
ZIP 容器的文件时间等信息可能使新 ZIP 哈希不同，结果内容以 CSV 哈希为准。
更换硬件、依赖版本或计算批量时，不预先保证逐位一致；入口会据实际结果报告。

## 从头复现该模型的原训练方案

将原始训练包放到 `code/train.zip`，在匹配的 Linux/CUDA 环境执行：

```bash
python code/reproduce_submitted_train.py
```

入口先检查训练包校验值。第一阶段 4096 步；第二阶段保留第一阶段优化器状态，
先到 1536 步再续至 2048 步，学习率调度长度 3072。
训练结束的模型在 `code/runs/stage2/model.pt`，归档的原提交模型仍在 `model/model.pt`。
复现训练模型推理时需显式用 `predict.py --model` 指定；固定版本入口默认绑定原提交权重。
训练缓存两种共约 141.9GB；若同时生成训练与测试缓存，建议准备约 260GB 可用空间。

官方初始化来自 `openai/clip-vit-base-patch32`，固定 revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，由 `fetch_weights.py` 记录并校验。
`stage1_model.pt` 仅供归档；四视图推理只加载 `model/model.pt`，不加载旧复赛权重或其他模型投票。
