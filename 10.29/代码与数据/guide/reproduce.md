# 操作入口与阶段续接

## 安装与位置

记录环境在 `settings/environment.json`，固定推理参数在 `settings/inference.json`。安装CUDA版本匹配的PyTorch后安装requirements。完整模型、代码目录及结果ZIP保持官方四项之间的相对位置；`scene.sh`从自己的位置解析目录，可从其他工作目录调用。

## 完整查询

`bash scene.sh inspect`先核对固定模型和结果。`infer --output /path/to/new_prediction`要求目标目录不存在，依次产生四个采集分片，CPU合并到bank，再完成固定图传播。输入TEST目录必须包含随包清单中全部49,857张JPG。

如果已有部分成功分片，显式执行 `infer --stage collect --part K --output /path/to/new_prediction`，只运行缺少的K（0至3）。四片齐全后用 `CUDA_VISIBLE_DEVICES='' bash scene.sh infer --stage merge --output ...` 合并，再用 `--stage readout`生成预测。续接采用本版本代码生成的分片，采集记录绑定源码及完整模型。

已有完整bank时可在Python中调用 `query.predict.readout(bank, output, cache)`，三项均为目录路径，输出目录内不得已有CSV或ZIP。只读出须设置 `CUDA_VISIBLE_DEVICES=''`。该操作先核对bank文件SHA256和完整文件名顺序，再计算预测。结果检验用 `scene.sh inspect --prediction .../pred_results.csv`。

## 重建训练输入

`scene.sh init`下载固定官方CLIP revision，核对权重SHA256。`scene.sh cache --train-dir ... --cache ...`逐图检查TRAIN原文件和清单，重建RGB短边BICUBIC缩放、中心512画布。像素缓存SHA256须为 `943a8e6d79d339e95bd77f96e85ac517f89785b70fd42e748081061781474c7d`，约66.1 GiB；随包官方224特征shape=(90197,512)、float16。缓存重建使用CPU图像预处理。

## 训练和继续阶段

`scene.sh learn --dry-run`打印实际预算，不进行训练。完整训练 `--stage all`依次调用initial、alignment和refine；可在已完成前阶段后单独指定后续阶段。保持同一 `--output`、`--cache`和 `--clip-dir`参数。默认阶段路径：

| 子目录 | 必需父模型 | 完成产物 |
| --- | --- | --- |
| `initial/` | 官方初始化 | `scene_final.pt`，累计10轮 |
| `alignment/` | 上一行模型及scene_receipt | `alignment_final.pt`，累计12轮 |
| `refine/` | 上一行模型及alignment_receipt | `regional_final.pt`，累计14轮，route=0 |

每阶段使用新的目标子目录，存在即停止；后续阶段核对父模型的完成标志、轮数、步数、哈希、原始投影及源码绑定。本入口支持阶段边界续接。阶段内部的优化器状态另行保存，入口不会自动恢复它。每个追加阶段按固定规则重置随机数和优化器。

训练batch=88、640输入，原训练峰值分配约13.3 GiB；建议训练与缓存磁盘至少90 GiB。GPU入口检查所选设备并独占相应锁。完整14轮重训按新运行记录；包内模型的原图固定八输入预测已完成并与获分CSV逐字节一致。
