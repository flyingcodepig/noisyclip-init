# 多代理模块任务卡

## 通用交付要求

将某个任务卡交给新代理时，附上以下统一要求：

> 只实现任务卡列出的所有权文件；不得修改公共接口名称。先阅读 `01_code_architecture` 全部文档。实现必须包含类型标注、docstring、单元测试、最小配置示例和错误路径测试。禁止访问测试集训练数据，禁止引入其他视觉骨干或外部数据。若接口不足，先提出兼容扩展建议，不得直接破坏契约。交付时报告修改文件、测试命令、已知限制和未解决风险。

## Agent A：配置、数据索引与合规边界

**所有权**：`config/`、`data/`、相关单测。

**输入**：训练根目录、测试图片列表、随机种子、验证比例。

**输出**：`class_to_idx.json`、train/val/test manifests、摘要和泄漏检查报告。

**必须实现**：

- 字典序读取四位类别目录，不假设连续编号；检查唯一性和格式。
- Pillow 可读取性、尺寸、文件哈希和稳定 `sample_id`。
- 固定分层划分；相同 seed 产生完全相同清单。
- 路径与哈希交叉检查，阻止 test 进入 train/val。
- 原始数据目录只读检查；任何清洗结果只写派生 manifest。

**验收**：同数据同 seed 的 digest 相同；修改一张文件后 digest 改变；故意把 test 路径加入 train 时预检失败。

## Agent B：CLIP、分类头与 LoRA

**所有权**：`models/`、模型相关单测。

**必须实现**：

- 只允许 `ViT-B/32` 与 `openai` 官方权重标识。
- 记录下载 URL/库版本/文件 SHA256；离线加载失败时明确报错。
- Linear 和 Cosine 两种 head；温度参数边界可配置。
- LoRA 仅注入配置指定层；默认后 4 块 attention q/v，rank=8。
- 输出可训练参数清单；发现未授权主干参数可训练时立即失败。
- 合并 LoRA 后导出单模型，验证合并前后输出等价。

**验收**：B0 backbone 梯度为零；B2 只有 head 和 LoRA 有梯度；导出模型不依赖 PEFT 运行时也能加载。

## Agent C：样本状态、原型与可信度

**所有权**：`noise/` 和 `models/prototypes.py`。

**必须实现**：

- 单原型的普通均值、截断均值、样本权重均值。
- EMA loss、增强一致性、原型相似度、原型 margin、预测稳定性。
- 每类内部百分位/rank 归一化；禁止全局阈值作为默认值。
- 连续监督权重和 trusted/uncertain/suspicious 分区。
- 状态以 `sample_id` 关联并原子提交；支持断点恢复。

**验收**：打乱 DataLoader 后状态仍与原样本对应；缺类和重复 `sample_id` 必须失败；所有权重有限且在 `[0,1]`。

## Agent D：损失函数

**所有权**：`losses/`。

**必须实现**：

- 加权 CE、标签平滑可开关。
- ELR 逐样本历史目标与 warmup 开关。
- weak/strong KL 或交叉熵一致性，target 端 detach。
- frozen teacher feature cosine anchor。
- CompositeLoss 按名称记录每个分量和有效样本权重。

**验收**：各损失独立关闭后 total 正确；全零监督权重时明确失败或执行配置规定的无监督路径；NaN/Inf 触发异常。

## Agent E：训练器、checkpoint 与指标

**所有权**：`engine/`、`metrics/`、`tracking/`。

**必须实现**：

- AMP、梯度累积、裁剪、scheduler、早停和统一日志。
- checkpoint 包含模型、优化器、scheduler、scaler、epoch、global_step、RNG 和样本状态版本。
- 固定验证集上的 Top-1、Macro、每类、后四分位、可信子集、特征漂移。
- 唯一运行目录、原子 checkpoint、`DONE/FAILED` 标记。
- 两 batch smoke test 和中断恢复一致性测试。

**验收**：从 epoch N 恢复后的下一步 loss 与不中断运行在容差内一致；磁盘空间不足时在保存前失败并保留旧 checkpoint。

## Agent F：推理、导出与提交

**所有权**：`submission/`、`cli/infer.py`、`cli/export.py`。

**必须实现**：

- 仅加载一个导出模型；默认单中心裁剪推理。
- 预测内部索引严格映射回原始 `class_id`。
- CSV 两列：图片文件名、四位类别编号；是否包含 header 由配置控制。
- 检查行数、重复、缺失、多余文件、扩展名大小写和类别合法性。
- 模型包包含配置、映射、预处理规范和权重哈希。

**验收**：故意交换类别顺序、漏一张图、重复文件名、输出 `1` 而非 `0001` 时全部失败。

## Agent G：上限模块

**所有权**：只新增独立模块，不修改主线默认行为。

候选任务必须分别派发：

- G1：严格门控软伪标签。
- G2：课程学习日程。
- G3：每类多原型和分配策略。
- G4：条件式 logit adjustment。
- G5：TrustCLIP 风格梯度约束。
- G6：JoAPR 风格自适应分区。

每个模块必须提供 `enabled: false` 默认值、单因素配置、消融指标和资源开销说明。禁止一个代理同时把多个上限模块硬编码进训练器。

## 集成负责人检查表

- 所有代理是否基于同一 `INTERFACES.md`？
- 是否出现新的重复入口或隐藏配置？
- 是否有模块读取绝对数据路径或测试目录？
- 是否存在未经配置控制的默认算法行为变化？
- 是否新增单元测试和至少一个错误路径测试？
- 是否更新配置 schema 和文档？
- 是否保持 B0 行为不变？

