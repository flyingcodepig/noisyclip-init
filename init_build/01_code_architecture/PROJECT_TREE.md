# 目标代码仓库结构

后续具体实现应在项目根目录形成以下结构。目录名和公共入口属于稳定 API，代理不得自行改名。

```text
project/
  pyproject.toml
  README.md
  src/noisyclip/
    __init__.py
    cli/
      audit_data.py          # 构建清单、检查图片、生成固定划分
      train.py               # 唯一训练入口
      evaluate.py            # 验证集评估入口
      infer.py               # 测试集单模型推理
      export.py              # 合并 LoRA 并导出
      validate_submission.py # 提交文件校验入口
    config/
      loader.py              # Hydra/OmegaConf 解析与冻结
      schema.py              # Pydantic 配置模型
      invariants.py          # 跨字段规则，例如 test 不得用于训练
    data/
      records.py             # SampleRecord、Batch
      catalog.py             # 扫描类别目录并固定类别映射
      image_io.py            # Pillow 安全读取、截断图片处理
      manifests.py           # parquet/json 清单读写与摘要
      split.py               # 分层、确定性划分
      dataset.py             # 仅根据 manifest 读样本
      transforms.py          # train weak/strong、val/test 变换
      sampler.py             # 普通/温和类别采样
      leakage.py             # 路径、文件名、哈希交叉集合检查
    models/
      outputs.py             # ModelOutput
      clip_loader.py         # 官方权重来源、哈希和预处理审计
      backbone.py            # 图像编码统一接口
      classifier.py          # Linear/Cosine 分类头
      prototypes.py          # 单原型/多原型构建
      lora.py                # 注入、冻结、统计、合并
      student.py             # Backbone + Head 的统一模型
      teacher.py             # 训练期冻结 teacher
      export.py              # 推理模型打包
    noise/
      state.py               # SampleState、SampleStateStore
      signals.py             # EMA loss、一致性、原型 margin 等
      normalize.py           # 类内 rank/robust z-score
      trust.py               # 信号聚合为 trust_score/weight
      partition.py           # trusted/uncertain/suspicious 分区
      curriculum.py          # 各 epoch 样本权重日程
      pseudolabel.py         # 严格门控、软目标生成
    losses/
      outputs.py             # LossOutput
      weighted_ce.py
      elr.py
      consistency.py
      feature_anchor.py
      composite.py           # 按配置组装并记录各分量
    engine/
      context.py             # RunContext/EpochContext
      seed.py                # 确定性配置
      precision.py           # AMP、梯度缩放
      trainer.py             # 训练状态机
      evaluator.py           # Top-1/Macro/分组指标
      callbacks.py           # 状态更新、早停、日志
      checkpoint.py          # 原子保存、恢复、版本迁移
      distributed.py         # 可选 DDP；单 GPU 也走相同接口
    metrics/
      classification.py      # top1、macro、per-class
      robustness.py          # trusted/suspicious、增强一致率
      drift.py               # 与冻结 CLIP 特征余弦相似度
      calibration.py         # 可选 ECE/Brier
    tracking/
      manifest.py            # run manifest 和完成状态
      logger.py              # JSONL + console
      artifacts.py           # 统一产物路径，禁止随意写文件
      environment.py         # 记录系统/依赖/GPU/代码版本
    submission/
      mapping.py             # internal index -> 原始四位 class_id
      writer.py              # 预测 CSV
      validator.py           # 行数、文件名、类别格式、重复项
    utils/
      atomic.py
      hashing.py
      paths.py
  tests/
    unit/
      test_class_mapping.py
      test_split_determinism.py
      test_cosine_head.py
      test_lora_freeze.py
      test_trust_signals.py
      test_elr_state.py
      test_pseudolabel_gate.py
      test_submission_validator.py
    integration/
      test_two_batch_train.py
      test_checkpoint_resume.py
      test_export_equivalence.py
      test_no_test_data_in_training.py
    fixtures/
      tiny_dataset/
  configs/
    base.yaml
    experiment/
    server/
    paths.local.yaml          # 不提交；服务器实际路径
  scripts/
  runs/                       # 不提交
```

## 依赖方向

```text
config, utils
    ↑
data, models, noise, losses, metrics
    ↑
engine, tracking, submission
    ↑
cli
```

禁止事项：

- `data` 不得导入 `engine`。
- `models` 不得读取 YAML、环境变量或磁盘路径。
- `losses` 不得直接更新 `SampleStateStore`。
- `noise` 不得读取测试集或推理结果。
- `cli` 不得包含算法实现，只负责组装、校验和调用。
- 任何模块不得直接在当前目录创建结果文件，必须通过 `ArtifactStore`。

## 入口命令约定

```bash
python -m noisyclip.cli.audit_data --config configs/base.yaml
python -m noisyclip.cli.train --config configs/experiment/b3_trust_weight.yaml
python -m noisyclip.cli.evaluate --run-dir runs/<run_id>
python -m noisyclip.cli.export --run-dir runs/<run_id> --output exported/model.pt
python -m noisyclip.cli.infer --model exported/model.pt --config configs/infer.yaml
python -m noisyclip.cli.validate_submission --csv predictions/pred_results.csv
```

所有入口成功返回 0；配置错误返回 2；数据边界/合规错误返回 3；训练失败返回 4；产物校验失败返回 5。

