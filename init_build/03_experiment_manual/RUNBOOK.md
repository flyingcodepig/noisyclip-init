# 单次实验标准操作流程

## A. 实验申请

1. 从 `experiment_registry.csv` 申请唯一实验 ID。
2. 写出唯一研究问题，例如“类内可信度连续加权是否降低噪声记忆并提升 Top-1？”
3. 指定基准 run 和唯一变化项。
4. 预先写明成功标准、失败标准和停止条件。
5. 估算显存、训练时长和磁盘占用。

如果无法用一句话说明唯一变量，不允许启动。

## B. 训练前检查

### 数据

- 原始 train/test 目录均为只读挂载。
- train/val/test manifest 路径和哈希无交集。
- 训练与验证 manifest digest 与比较基准相同。
- `class_to_idx.json` 相同，包含500个唯一四位字符串。
- 验证集文件清单已固定，不因实验重新随机划分。
- 不存在人工 clean list、外部类别名称或外部数据路径。

### 代码与配置

- 工作区代码版本已记录；有未提交修改时保存 diff 摘要。
- resolved config 能通过 schema 和跨字段检查。
- 与基准配置 diff 只包含申明的唯一变量。
- checkpoint 恢复时，模型结构、类别映射、优化器和样本状态版本一致。
- 官方 CLIP 权重来源和 SHA256 与 allowlist 一致。

### 资源

- `nvidia-smi` 正常，无其他训练占用目标 GPU。
- 剩余显存满足配置估算。
- run 盘剩余空间不低于预计产物的2倍。
- 当前 run 目录不存在。
- SSH 会话已置于 tmux/调度器中。

### 冒烟测试

```text
1 batch forward
1 batch backward
2 optimizer steps
1 small validation
1 checkpoint save
1 checkpoint reload
1 inference CSV fragment
```

检查 loss 有限、梯度只出现在允许参数、checkpoint 可恢复、类别映射正确后，才开始完整训练。

## C. 启动实验

1. 执行环境快照脚本。
2. 执行 preflight 并保存报告。
3. 启动训练，将 stdout/stderr 同时写入 run 日志。
4. 在实验登记表记录主机、GPU、PID/作业 ID、启动时间。
5. 不在运行中修改 YAML；需要修改时停止并创建新实验 ID。

## D. 训练中监控

每轮自动记录：

- total/CE/ELR/consistency/feature-anchor loss；
- 学习率、梯度范数、AMP scaler；
- Top-1、Macro、困难类别、可信子集指标；
- trusted/uncertain/suspicious 比例和平均权重；
- 预测类别分布、熵和增强一致率；
- 与原始 CLIP 的特征余弦相似度；
- GPU 利用率、峰值显存、吞吐、磁盘剩余空间。

需要人工查看但不得手工干预训练状态：

- loss 是否突然 NaN/爆炸；
- 所有预测是否塌缩到少数类别；
- suspicious 比例是否快速接近 0 或 1；
- 伪标签是否由少数类别垄断；
- train accuracy 上升而可信验证集连续下降；
- feature cosine 是否突然大幅下降；
- DataLoader 是否大量报告损坏图片。

触发 `RISK_REGISTER.md` 的停止条件时，执行安全停止，不要让程序继续覆盖最后有效 checkpoint。

## E. 训练后评估

1. 确认 run 有 `DONE`，不存在未处理异常。
2. 从磁盘重新加载 `best_top1.pt`，重新跑完整固定验证集。
3. 同样重新评估 `last.pt`，分析后期噪声记忆。
4. 导出每类准确率、混淆矩阵、置信度分布、分组指标。
5. 运行 checkpoint 完整性和类别映射校验。
6. 若是 LoRA 实验，验证合并前后输出等价。
7. 写一段结论：是否达到预定义成功条件、收益来自哪些类别、代价是什么。

## F. 复验流程

单种子初筛通过后：

1. 不改变任何超参数，只更换第二个预登记种子。
2. 若两个种子方向一致，再运行第三个种子或第二个固定划分。
3. 计算均值、标准差、相对基准的配对差值。
4. 检查 Macro、后四分位类别、可信子集和特征漂移是否存在反向恶化。
5. 根据 `DECISION_RULES.md` 决定准入、待定或淘汰。

## G. 最终全数据重训

只有最终模块组合冻结后才能进行：

1. 固定代码版本、容器镜像 digest 和全部配置。
2. 将 train+val 恢复为官方全部训练数据；测试仍不参与任何训练。
3. 根据主线实验选择固定 epoch 数或不依赖验证标签的停止策略。
4. 重新计算全部训练样本原型和可信度，不能复用旧 split 的样本状态。
5. 完成单模型导出、离线推理和 CSV 校验。
6. 保存训练报告、模型哈希、类别映射、预处理配置和提交文件哈希。

