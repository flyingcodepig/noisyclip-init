# 初赛代码架构总说明

## 1. 目标与边界

本目录定义初赛 500 类噪声标签细粒度图像识别项目的代码契约。它不是完整实现，而是供多个实现代理共同遵循的唯一架构基线。任何代理开始编码前，必须先阅读本文件、`PROJECT_TREE.md`、`INTERFACES.md`、`MODULE_ASSIGNMENTS.md` 和 `QUALITY_GATES.md`。

项目必须满足以下不可破坏约束：

1. 骨干网络只能是官方 OpenAI CLIP ViT-B/32 预训练权重。
2. 训练、验证、噪声筛选、原型计算只能使用当前阶段官方训练集。
3. 测试集只能推理，禁止进入训练、阈值估计、自监督或伪标签流程。
4. 类别 ID 以目录字符串为准，例如 `0001`；禁止假设连续、禁止自行 `+1/-1`。
5. 原始数据只读；所有派生清单、状态和模型写入独立运行目录。
6. 最终推理只加载一个导出模型；训练期 teacher 不进入推理图。
7. 所有人工判断必须转化为可配置、可复现的代码规则，禁止人工清洗名单成为训练前置条件。

## 2. 推荐实现主线

```text
数据审计与分层划分
  -> 冻结 CLIP + 余弦分类头基线
  -> 类别鲁棒原型
  -> 类内样本可信度建模
  -> 后层 attention q/v LoRA
  -> 加权 CE + ELR
  -> 弱/强增强一致性
  -> 冻结 CLIP 特征保持
  -> 可选后期软伪标签
  -> 合并 LoRA，导出单模型
  -> 生成并校验 CSV
```

## 3. 架构原则

- **配置驱动**：实验差异只能通过 YAML 配置表达，不允许复制训练脚本形成 `train_v2.py`、`train_final.py`。
- **接口稳定**：模块只通过 `INTERFACES.md` 中的数据类和协议交互。
- **单向依赖**：`data -> models/noise/losses -> engine -> cli`，底层模块不得反向导入训练入口。
- **运行不可覆盖**：每次运行生成唯一 `run_id`；存在同名目录时必须失败，不能覆盖。
- **可追溯**：配置、代码版本、环境、数据清单摘要、随机种子、指标、checkpoint 必须绑定同一 `run_id`。
- **先验证后训练**：数据边界、类别映射、权重来源、GPU、磁盘空间和输出目录检查失败时，不得启动训练。
- **状态显式化**：样本可信度、EMA loss、伪标签等状态存储在 `SampleStateStore`，不得隐藏在 Dataset 或全局变量中。
- **模块可关闭**：LoRA、ELR、一致性、特征保持、伪标签、类别均衡都必须可独立关闭，以支持消融。

## 4. 四阶段交付定义

### Phase A：基础链路

实现数据索引、固定划分、CLIP 加载、分类头、训练循环、评估、checkpoint、推理和提交校验。对应 B0/B1。

### Phase B：参数高效适配

实现 LoRA 注入、参数冻结审计、可训练参数统计、LoRA 合并与单模型导出。对应 B2。

### Phase C：抗噪主线

实现样本状态、原型、可信度信号、连续权重、ELR、特征保持和一致性。对应 B3-B6。

### Phase D：上限模块

实现软伪标签、课程学习、多原型、条件式长尾校正、JoAPR/TrustCLIP 风格模块。每个模块必须在配置中默认关闭。

## 5. 标准运行产物

每次训练必须产生：

```text
runs/<run_id>/
  resolved_config.yaml
  manifest.json
  environment/
    pip_freeze.txt
    nvidia_smi.txt
    git_state.txt
  data/
    class_to_idx.json
    train_manifest.parquet
    val_manifest.parquet
    manifest_digest.json
  metrics/
    epoch_metrics.jsonl
    best_metrics.json
    per_class_metrics.csv
  sample_state/
    epoch_XXXX.parquet
  checkpoints/
    last.pt
    best_top1.pt
    best_macro.pt
  artifacts/
    prototypes.pt
    confusion_matrix.npy
  logs/
    train.log
  DONE | FAILED
```

`DONE` 只能在训练、最终评估、产物完整性检查全部成功后创建；异常退出必须写 `FAILED`，包含错误类型和最后完成阶段。

## 6. 配套文档

- `PROJECT_TREE.md`：目标代码仓库目录结构和依赖方向。
- `INTERFACES.md`：统一 Python 数据结构、协议、张量形状和生命周期。
- `MODULE_ASSIGNMENTS.md`：可直接复制给新代理的任务卡和验收标准。
- `QUALITY_GATES.md`：合并前必须通过的静态检查、单测、集成测试和合规检查。

