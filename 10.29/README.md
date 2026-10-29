# 2026-10-29 代码与模型快照

本目录保存本地 `10.29` 文件夹中的代码、运行配置和模型，保持原文件名与相对路径。仓库原有实现继续保留在根目录。`inputs/` 数据目录不属于此次上传范围。

## 获取完整模型

需要安装 Git LFS；仅下载 GitHub 源码 ZIP 可能无法取得实际大文件。

```bash
git clone --branch release/2026-10-29-code-model https://github.com/flyingcodepig/noisyclip-init.git
cd noisyclip-init
git lfs install
git lfs pull
```

模型：`AIC-2026-12985846-模型文件.pt`（352,491,173 字节）。

SHA-256：`95a286d031e4d95085d7ba7bf78f17a7a55c3a51e0f7688474ca74c311031586`。

## 使用说明

- 完整操作说明：[guide/reproduce.md](代码与数据/guide/reproduce.md)。
- 模型与损失说明：[guide/model_and_losses.md](代码与数据/guide/model_and_losses.md)。
- 数据说明：[guide/data.md](代码与数据/guide/data.md)。
- 固定推理配置：[settings/inference.json](代码与数据/settings/inference.json)。
- 原模型成绩及校验记录：[checks/official_result.json](代码与数据/checks/official_result.json)。记录的成绩为 76.90671162491972，本次上传没有重新训练或评测。

原始 TRAIN/TEST 图片、可重建像素缓存及结果 ZIP 不在此次源目录中；源目录的 `inputs/`（训练清单、测试文件名、官方特征和训练轨迹数据）此次未上传。复现前需通过另行授权的数据渠道补齐这些输入。`scene.sh inspect` 的完整结果检查还需要同层的 `AIC-2026-12985846-赛题结果文档.zip`；使用该命令前需另行补齐，不应跳过校验。

模型通过 Git LFS 保存。`upload_manifest.json` 仅记录此次上传的源文件尺寸和 SHA-256，便于下载后复核。
