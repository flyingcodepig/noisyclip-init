# 初赛实验手册

## 1. 手册目标

本手册规定一次实验从申请编号、训练前检查、启动、监控、评估、复验到准入主线的完整流程。任何实验不得只凭一次最佳 Top-1 宣布有效。

核心原则：

1. 先完成 B0-B6 主线消融，再开展 U 系列上限实验。
2. 同一比较组使用相同数据划分、种子、训练预算和评价脚本。
3. 单因素筛选，确认有效后才做组合实验。
4. 验证集来自带噪训练数据，因此同时观察 Top-1、Macro、可信子集、困难类别、特征漂移和多种子稳定性。
5. 测试集和线上成绩不能用于训练、阈值估计或逐轮选择。
6. 实验记录和模型产物不完整的 run 不进入比较表。

## 2. 文档导航

- `END_TO_END_BEGINNER_GUIDE.md`：从本机代理开发、Git交付、服务器部署到最终提交的完整新手流程，并在每个阶段嵌入反馈检查点。
- `RUNBOOK.md`：每次实验都必须执行的标准操作步骤。
- `EXPERIMENT_CATALOG.md`：B0-B6 和 U1-U6 的目的、变量、步骤、记录与停止条件。
- `RECORDS_AND_ARTIFACTS.md`：指标、运行清单、样本状态和 checkpoint 的保存规范。
- `DECISION_RULES.md`：如何判断模块进入主线、复验或淘汰。
- `RISK_REGISTER.md`：可能造成无效实验、数据泄漏、权重损坏或灾难性后果的步骤。

第一次执行项目时，请先完整阅读 `END_TO_END_BEGINNER_GUIDE.md`，再按 `RUNBOOK.md` 开始单次实验。

## 3. 实验编号

```text
<stage>-<module>-<seed>-<attempt>
```

示例：

```text
B3-trust-s20260806-a01
U1-pseudo-s20260806-a02
```

禁止使用 `final`、`best_new` 等不可追踪名称。一个 run ID 对应唯一 resolved config、代码版本、数据 digest 和随机种子。

## 4. 实验状态

```text
PLANNED -> PREFLIGHT -> RUNNING -> EVALUATED -> REVIEWED
                               -> FAILED
                               -> STOPPED_SAFELY
```

- `FAILED`：程序、硬件、数据或产物错误，结果无效。
- `STOPPED_SAFELY`：按照预定义停止条件终止，可以用于风险分析，但不能与完整训练直接比较。
- `REVIEWED`：完成指标核对、产物校验和书面结论。

## 5. 主线冻结点

完成 B0-B6 后创建主线冻结记录，至少写明：

- 采用到哪一个 B 版本；
- 所有已启用模块及配置摘要；
- 选择的训练预算和最佳 checkpoint 规则；
- 两个以上种子的均值、标准差；
- 固定验证 manifest digest；
- 哪些模块被淘汰以及原因。

U 系列实验只能相对于这个冻结主线增加一个变量。
