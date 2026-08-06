# 统一接口与数据契约

本文件中的公共名称、字段、类型和张量形状是跨模块契约。实现可以扩展字段，但不得改变既有字段语义。

## 1. 基本数据类

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str          # 稳定 ID：相对路径的 SHA256，不含绝对服务器路径
    relative_path: str      # 相对 official train/test 根目录
    split: str              # train | val | test
    class_id: str | None    # 原始四位目录名；test 为 None
    target: int | None      # 内部 [0, C)；test 为 None
    file_sha256: str | None
    width: int | None
    height: int | None
    readable: bool

@dataclass(slots=True)
class Batch:
    sample_ids: list[str]
    image_weak: Tensor      # float32/AMP, [B, 3, 224, 224]
    image_strong: Tensor | None
    targets: Tensor | None  # int64, [B]
    class_ids: list[str] | None

@dataclass(slots=True)
class ModelOutput:
    logits: Tensor          # [B, C], 未 softmax
    embedding: Tensor       # [B, D], L2-normalized
    temperature: Tensor | None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

@dataclass(slots=True)
class SampleState:
    sample_id: str
    seen_count: int
    ema_loss: float
    ema_probs: list[float] | None
    prediction_stability: float
    augmentation_agreement: float
    prototype_similarity: float
    prototype_margin: float
    trust_score: float      # [0, 1]
    supervised_weight: float # [0, 1]
    partition: str          # trusted | uncertain | suspicious
    pseudo_target: int | None
    pseudo_confidence: float | None
    updated_epoch: int

@dataclass(slots=True)
class LossOutput:
    total: Tensor           # scalar
    components: Mapping[str, Tensor] # 每个值必须是 scalar
    per_sample_supervised: Tensor | None # [B]，供状态更新，必须 detach 后消费

@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    seed: int
    num_classes: int
    class_to_idx: Mapping[str, int]
    config_digest: str
    data_digest: str
```

## 2. 模型协议

```python
from typing import Protocol

class ImageEncoder(Protocol):
    embedding_dim: int
    def encode_image(self, images: Tensor) -> Tensor:
        """返回 [B, D] L2-normalized 图像特征。不得返回 logits。"""

class ClassifierHead(Protocol):
    num_classes: int
    def __call__(self, embeddings: Tensor) -> Tensor:
        """输入 [B, D]，返回未 softmax 的 [B, C] logits。"""

class StudentModel(Protocol):
    def forward(self, images: Tensor) -> ModelOutput: ...
    def trainable_parameter_report(self) -> dict[str, int | float]: ...
    def export_single_model(self, destination: Path) -> Path: ...

class FrozenTeacher(Protocol):
    @torch.inference_mode()
    def encode_image(self, images: Tensor) -> Tensor: ...
```

模型约束：

- `FrozenTeacher` 构造后必须 `eval()` 且全部 `requires_grad=False`。
- `StudentModel.embedding` 必须归一化；特征保持损失以该接口为准。
- B0/B1 时 backbone 全冻结；B2 之后只有显式配置的 LoRA 和 head 可训练。
- 导出前后同一输入 logits 最大绝对差应小于 `1e-5`（FP32）或配置容差（AMP）。

## 3. 原型与可信度协议

```python
class PrototypeBuilder(Protocol):
    def fit(
        self,
        embeddings: Tensor,       # [N, D]
        targets: Tensor,          # [N]
        sample_weights: Tensor | None,
        num_classes: int,
    ) -> Tensor:
        """返回 [C, D] L2-normalized 原型；缺类必须报错。"""

class TrustSignal(Protocol):
    name: str
    def compute(
        self,
        batch: Batch,
        output_weak: ModelOutput,
        output_strong: ModelOutput | None,
        state: list[SampleState],
        prototypes: Tensor | None,
    ) -> Tensor:
        """返回 [B] 原始信号；不在此处归一化或落盘。"""

class TrustAggregator(Protocol):
    def update_epoch(
        self,
        records: list[SampleRecord],
        raw_signals: Mapping[str, Tensor],
        previous: list[SampleState],
        epoch: int,
    ) -> list[SampleState]:
        """必须按 class_id 类内归一化，再聚合并产生连续权重。"""

class SampleStateStore(Protocol):
    def load(self, sample_ids: list[str]) -> list[SampleState]: ...
    def stage_epoch(self, states: list[SampleState], epoch: int) -> Path: ...
    def commit_epoch(self, epoch: int) -> None: ...
    def rollback_uncommitted(self) -> None: ...
```

状态更新必须事务化：先写临时文件，校验样本数、唯一性、数值范围后原子替换。训练异常时不得留下半个 epoch 的状态。

## 4. 损失协议

```python
class LossTerm(Protocol):
    name: str
    def __call__(
        self,
        batch: Batch,
        student_weak: ModelOutput,
        student_strong: ModelOutput | None,
        teacher_embedding: Tensor | None,
        sample_states: list[SampleState],
        epoch: int,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """返回 scalar；若需逐样本状态，同时返回 detached 前的 [B] loss。"""

class CompositeLoss(Protocol):
    def __call__(...) -> LossOutput: ...
```

损失组合约束：

- 交叉熵必须使用 `supervised_weight`，并以权重和归一化，避免 batch 中低权重样本过多导致整体梯度缩小。
- ELR 状态只能由 `sample_id` 关联，不得依赖 DataLoader 顺序。
- 一致性 teacher target 必须 `detach()`，否则产生双向塌缩梯度。
- feature anchor 的 teacher 特征必须在 `no_grad` 下计算。
- 任何 loss 产生 NaN/Inf 必须立即中止，不得静默跳过 batch。

## 5. 训练状态机

```text
CREATED
  -> PREFLIGHT_OK
  -> DATA_READY
  -> MODEL_READY
  -> TRAINING
  -> VALIDATING
  -> CHECKPOINTED
  -> COMPLETED

任意状态 -> FAILED
```

训练器每个 epoch 的顺序固定为：

1. 载入上一个已提交的 `SampleState`。
2. 设置课程学习日程和 loss 权重。
3. 训练一个 epoch，记录逐样本原始信号。
4. 在固定验证清单上评估。
5. 聚合并校验新样本状态。
6. 原子保存 checkpoint、optimizer、scheduler、scaler、RNG 状态。
7. 提交新样本状态。
8. 写入 epoch JSONL 记录。

checkpoint 恢复必须恢复 Python、NumPy、PyTorch CPU/CUDA RNG，确保断点前后数据顺序和权重更新一致。

## 6. 指标协议

每轮至少输出：

```json
{
  "epoch": 12,
  "train/loss_total": 1.23,
  "train/loss_ce": 0.91,
  "train/loss_elr": 0.12,
  "val/top1": 0.701,
  "val/macro_accuracy": 0.688,
  "val/bottom_quartile_accuracy": 0.421,
  "val/trusted_top1": 0.754,
  "val/augmentation_agreement": 0.861,
  "val/feature_cosine_to_base": 0.934,
  "model/trainable_parameters": 1245184,
  "system/max_gpu_memory_mib": 18342
}
```

所有比例统一存 `[0, 1]`，展示层才转换成百分数。指标缺失时写 `null` 和原因，禁止写 0 冒充未计算。

## 7. 配置接口

顶层配置固定包含：

```yaml
experiment: {}
paths: {}
data: {}
model: {}
noise: {}
loss: {}
trainer: {}
evaluation: {}
tracking: {}
submission: {}
```

配置加载后必须：解析继承 -> 环境变量替换 -> Pydantic 校验 -> 跨字段不变量检查 -> 写 `resolved_config.yaml` -> 冻结对象。训练过程中不得修改配置对象。

