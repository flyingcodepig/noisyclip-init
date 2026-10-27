# 71.0621387283237 分源码快照

此目录来自已校验的四视图分类归档，包含全部 33 个 Python 源文件、训练与推理配置、环境记录及原始 SHA256 清单。代码保持归档原样；不包含后续优化实验。

## 使用

仓库提交源码及小型配置文件。模型权重、数据索引、候选标签数组和原始预测结果保留在本地完整归档中，未上传到 GitHub。恢复这些文件后，按 README.md 执行训练或四视图推理。

在本目录运行（仅需 Python 标准库）：

```bash
python restore_runtime_artifacts.py "/path/to/AIC-2026-12985846-四视图分类版-代码与模型.zip"
python verify_package.py
```

本地原归档位于 `E:/aic复赛/deliveries/AIC-2026-12985846-四视图分类版-代码与模型.zip`，大小 1,071,630,768 字节，SHA256 为 `82d8c496089d87d118932e30cbe9b99ca4ecad90a777c5e5fe5534f362a1aadc`。原始 train.zip / test.zip 仍需另行准备。

恢复入口检查归档及每个文件的 SHA256；已有文件内容不同时停止。

未上传文件：

- `code/data/train_index.json`
- `code/data/test_names.json`
- `code/cache/train_candidate_labels.npy`
- `code/weights/clip-vit-base-patch32/pytorch_model.bin`
- `model/model.pt`
- `model/stage1_model.pt`
- `results/pred_results.csv`
- `results/pred_results.zip`
