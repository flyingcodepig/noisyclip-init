# 实验记录与产物规范

## 1. 实验登记表字段

```text
experiment_id
status
research_question
baseline_run_id
single_change
config_path
config_digest
code_revision
dirty_diff_digest
container_image_digest
data_manifest_digest
class_mapping_digest
seed
host
gpu_name
gpu_uuid
start_time
end_time
best_epoch
best_val_top1
best_val_macro
trusted_top1
bottom_quartile_accuracy
feature_cosine_to_base
peak_vram_mib
train_minutes
decision
notes
```

登记表不存储庞大日志，只存可搜索摘要和产物相对路径。

## 2. 每轮指标

`epoch_metrics.jsonl` 每行一个合法 JSON 对象，至少包含：

- epoch/global_step；
- 各损失分量；
- train/val Top-1；
- Macro 和每类准确率摘要；
- 后四分位类别准确率；
- trusted/uncertain/suspicious 数量和平均监督权重；
- 增强一致率、预测熵；
- feature cosine；
- 学习率、梯度范数、AMP scaler；
- 吞吐、峰值显存、耗时和剩余磁盘。

禁止只保存 TensorBoard event 而不保存机器可读 JSONL。

## 3. 样本级状态

每次可信度更新保存 parquet：

```text
sample_id
class_id
target
ema_loss
prediction_stability
augmentation_agreement
prototype_similarity
prototype_margin
trust_score
supervised_weight
partition
pseudo_target
pseudo_confidence
updated_epoch
```

样本状态可能暴露数据结构，只能保存在比赛项目私有存储中。

## 4. Checkpoint 内容

```python
{
  "format_version": 1,
  "run_id": str,
  "epoch": int,
  "global_step": int,
  "model": state_dict,
  "optimizer": state_dict,
  "scheduler": state_dict,
  "grad_scaler": state_dict,
  "rng": {
    "python": object,
    "numpy": object,
    "torch_cpu": Tensor,
    "torch_cuda": list[Tensor],
  },
  "sample_state_version": int,
  "class_mapping_digest": str,
  "data_manifest_digest": str,
  "resolved_config_digest": str,
  "metrics_at_save": dict,
}
```

恢复时任何 digest 不一致必须拒绝，除非运行专门、可审计的迁移工具。

## 5. 必须长期保存

- 最终纳入比较的所有 resolved config 和 manifest 摘要；
- 主线每阶段最佳 checkpoint；
- 每个候选模块至少一个最佳 checkpoint；
- 最终模型、类别映射、预处理配置和模型 SHA256；
- 所有实验指标 JSONL、per-class CSV 和比较表；
- 失败实验的错误日志与最后一个有效诊断 checkpoint；
- 容器镜像名称和 digest、依赖冻结文件。

## 6. 可以定期清理但需先确认

- 非最佳的中间 epoch checkpoint；
- 可重建的冻结特征缓存；
- Docker build cache；
- 未进入比较表的重复失败 run。

清理前生成删除候选清单，确认不包含唯一 checkpoint、最终提交、类别映射或实验记录。优先移入回收/归档区，保留至少7天。

