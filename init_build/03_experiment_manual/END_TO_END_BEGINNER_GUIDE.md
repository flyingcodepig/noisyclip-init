# 端到端新手流程与反馈检查点

## 1. 如何使用本流程

流程中的地点标记：

- **[本机]**：Windows 本机 `D:\My_Project\AIC`。
- **[服务器]**：租用的 Ubuntu GPU 服务器。
- **[代理]**：负责实现代码的新会话或子代理。
- **[反馈 Fxx]**：到达此处时，把指定信息反馈给主协作会话。
- **[门禁]**：未通过时禁止进入下一阶段。

总顺序：

```text
F00 规则和数据事实确认
  -> F01 本机Git仓库和项目骨架
  -> F02 各代码模块交付
  -> F03 本机集成版本冻结
  -> F04 服务器健康检查
  -> F05 数据审计
  -> F06 S0冒烟测试
  -> F07 B0基线
  -> F08 B1-B6逐项消融
  -> F09 主线冻结
  -> F10 U系列上限方法
  -> F11 多种子与第二划分复验
  -> F12 全数据最终训练
  -> F13 单模型导出、推理、CSV校验和提交
  -> F14 归档
```

任何阶段出现 `F-ERROR` 中的问题，立即停止当前流程并反馈错误，不要继续尝试后面的步骤。

---

## 2. F00：规则与数据事实确认

### [本机] 你要做什么

确认官方文件、数据包和提交说明中是否存在：

- 真实类别名称或 `0001 -> 物种名称` 映射；
- 官方 starter code；
- 官方 baseline 分数；
- 测试文件列表；
- 提交 CSV 样例和表头说明；
- 关于 EMA、TTA、checkpoint averaging、训练期 teacher 的补充规则。

只查看文件，不手工整理、移动或删除训练图片。

### [反馈 F00] 发给我

```text
训练集目录结构示例：
测试集目录结构示例：
类别是否只有数字编号：
是否存在官方真实类名映射：
类别编号从0000还是0001开始：
编号是否连续：
测试列表前3-5行：
提交样例前3-5行：
CSV是否有表头：
是否有官方baseline/starter code：
是否有新增规则说明：
```

### [门禁]

类别语义和提交格式没有确认前，可以开发通用代码，但不能最终确定 Prompt 路线和 CSV 格式。

---

## 3. F01：建立本机 Git 仓库和项目骨架

### [本机] 你要做什么

1. 建立私有 Git 仓库。
2. 配置根目录 `.gitignore`，排除数据、权重、run、cache、`.env` 和密钥。
3. 提交当前 `init_build`。
4. 让第一个代理实现：

```text
pyproject.toml
src/noisyclip包结构
tests结构
配置加载器
公共数据类与协议
CLI空入口
```

5. 运行基础检查：

```powershell
python -m pytest -q
ruff check src tests
mypy src
git status
```

### [反馈 F01] 发给我

```text
私有仓库是否建立：是/否
当前commit ID：
项目树路径：
pytest结果：
ruff结果：
mypy结果：
失败测试及完整错误：
代理新增/修改文件列表：
代理是否修改公共接口：
```

优先提供文本输出或本机文件路径，不要只发模糊截图。

### [门禁]

- 项目骨架不能导入；
- 配置无法加载；
- 测试入口不存在；
- `.gitignore` 没有排除数据和密钥；

出现任一情况都不能开始并行模块实现。

---

## 4. F02：代理分批实现代码模块

### 第一批代理

```text
代理A：config/ + data/
代理B：models/
代理F：submission/ + infer/export CLI
```

### 第二批代理

第一批合并并通过测试后再安排：

```text
代理E：engine/ + metrics/ + tracking/
代理C：noise/ + prototypes
代理D：losses/
```

### 第三批代理

主线 B0-B6 可运行后，再分别安排 U1-U6 上限模块。

### [本机] 每个代理交付后要做什么

1. 查看 `git status` 和 `git diff`。
2. 确认代理只修改任务所有权内的文件。
3. 运行该模块单元测试。
4. 运行已有全部测试，确认没有破坏旧功能。
5. 一个模块一个 Git commit。

### [反馈 F02] 每个代理都发一份

```text
代理任务名称：
实现模块：
新增/修改文件：
公共接口是否改变：
配置schema是否改变：
运行的测试命令：
通过/失败测试数量：
两批次smoke是否通过：
已知限制：
潜在合规风险：
git diff或commit ID：
```

如果代理要求改变 `INTERFACES.md` 中的公共接口，先反馈修改理由，不要直接让其他代理适配一个未经审核的新接口。

### [门禁]

- 新模块关闭后不能恢复原基线行为；
- 新模块没有配置开关；
- 没有单元测试；
- 读取测试集或外部数据；
- 修改不属于自己的模块且无说明；

出现任一情况，不合并该代理交付。

---

## 5. F03：冻结第一个可部署代码版本

### [本机] 你要做什么

确保至少已实现：

- 数据审计；
- CLIP加载；
- B0训练；
- 验证；
- checkpoint保存恢复；
- 推理；
- CSV校验；
- S0集成测试。

运行：

```powershell
ruff check src tests
ruff format --check src tests
mypy src
pytest -q tests/unit
pytest -q tests/integration
git status
```

全部通过后提交并推送：

```powershell
git add src tests configs pyproject.toml init_build
git commit -m "Prepare first server smoke-test release"
git push origin main
git rev-parse HEAD
```

### [反馈 F03] 发给我

```text
待部署commit ID：
单元测试：
集成测试：
配置校验：
当前仍存在的已知限制：
Dockerfile或requirements是否改变：
```

### [门禁]

测试失败或 Git 工作区存在未解释的修改时，不能把该版本作为服务器正式实验版本。

---

## 6. 本机代码如何部署到服务器

### 第一次部署

1. **[服务器]** 建立项目与数据目录。
2. **[服务器]** 从私有 Git 仓库克隆代码到 `/opt/noisyclip/project`。
3. **[本机/传输工具]** 使用 WinSCP、SFTP 或云盘把官方数据上传到 `/mnt/noisyclip-data/init`，不要上传到 Git。
4. **[服务器]** 配置 `init_build/02_deployment/.env`。
5. **[服务器]** 运行宿主机初始化、镜像构建和健康检查。

### 后续代码更新

本机代理交付并通过测试后：

```powershell
git add <本次文件>
git commit -m "清晰描述本次修改"
git push origin main
git rev-parse HEAD
```

记下 commit ID。确认服务器没有训练作业后：

```bash
cd /opt/noisyclip/project
git fetch origin
git switch --detach <本机commit-ID>
git rev-parse HEAD
```

服务器输出必须与本机 commit ID 一致。

- 只修改 `src/`、`tests/`、YAML：通常不需要重建镜像，重新跑测试即可。
- 修改 `Dockerfile`、`requirements.txt` 或系统依赖：必须重新 `--build` 并再次健康检查。
- 训练运行期间：绝对不能 `git pull`、`git switch` 或覆盖代码。

---

## 7. F04：服务器与容器健康检查

### [服务器] 你要做什么

```bash
nvidia-smi
cd /opt/noisyclip/project/init_build/02_deployment
bash bootstrap_ubuntu_24_04.sh --yes
cp .env.example .env
chmod 600 .env
nano .env
bash deploy.sh --env .env --build
bash deploy.sh --env .env --check
```

### [反馈 F04] 发给我

直接发送健康检查完整输出，并补充：

```text
服务器GPU型号/数量：
每张GPU显存：
CPU核数：
RAM：
NVMe容量和剩余空间：
驱动版本：
PyTorch版本：
CUDA版本：
Docker镜像是否构建成功：
train/test是否在容器内只读：
是否可以下载官方CLIP权重：
错误信息：
```

不要发送 `.env`、密码、SSH私钥、云密钥或访问令牌。

### [门禁]

GPU不可见、原始数据可写、运行盘不可写、空间不足或容器版本错误时，禁止开始数据审计和训练。

---

## 8. F05：数据审计

### [服务器] 你要做什么

在训练容器中运行数据审计入口。审计只生成 manifest 和报告，不移动、不删除原图。

### [反馈 F05] 发给我

```text
类别数量：
训练图片总数：
测试图片总数：
每类最少/最多/平均/中位样本数：
P10/P25/P75/P90类样本数：
不可读图片数量：
截断但可读图片数量：
完全重复图片数量：
跨类别重复图片数量：
异常尺寸图片数量：
train/val/test交集检查：
类别是否明显不均衡：
class_to_idx路径及digest：
manifest路径及digest：
审计报告路径：
```

若文件很多，不发送原始图片，只发送统计和报告路径。

### 我会据此判断

- 是否启用类别均衡；
- 是否调整验证划分；
- 是否需要多原型；
- 重复/损坏图片如何自动处理；
- 类内可信度最低样本保护阈值。

### [门禁]

类别不是500、映射不稳定、存在数据交集或图片读取策略未确定时，禁止进入S0。

---

## 9. F06：S0冒烟测试

### [服务器] 你要做什么

只运行极小数据、两个 optimizer step：

```text
读取batch
-> forward
-> backward
-> optimizer step
-> 验证
-> checkpoint保存
-> checkpoint恢复
-> 推理
-> CSV片段校验
```

### [反馈 F06] 发给我

```text
使用commit ID：
使用配置：
forward/backward是否成功：
初始和第二步loss：
loss/gradient是否全部有限：
实际有梯度的参数组：
backbone冻结审计：
teacher冻结审计：
checkpoint保存/恢复结果：
恢复前后输出最大差异：
LoRA合并前后输出最大差异：
峰值显存：
CSV校验结果：
完整错误栈：
run目录：
```

### [门禁]

S0任何步骤失败，都禁止启动B0完整训练。

---

## 10. F07：B0冻结CLIP基线

### [服务器] 你要做什么

运行完整 B0。训练过程中不要修改代码和配置。

### [反馈 F07] 发给我

```text
实验ID/run ID：
commit ID：
resolved config路径及digest：
data manifest digest：
随机种子：
最佳epoch/最后epoch：
最佳Top-1/最后Top-1：
Macro Accuracy：
后25%类别准确率：
训练-验证差距：
每epoch时间/总训练时间：
峰值显存：
backbone是否始终冻结：
最佳checkpoint路径：
per-class指标路径：
异常或警告：
```

### 我会据此判断

- 基线是否可信；
- 服务器训练预算；
- 学习率、epoch和早停是否合理；
- 哪些类别天然困难；
- 后续提升是否超过自然波动。

### [门禁]

B0未稳定完成，不能用B1-B6的结果证明算法提升。

---

## 11. F08：B1-B6逐项消融

严格顺序：

```text
B1 Cosine Head + 原型
B2 后层LoRA
B3 类内可信度连续加权
B4 ELR
B5 冻结CLIP特征保持
B6 弱强增强一致性
```

每次只相对上一个实验增加一个模块。

### [反馈 F08] 每个实验都使用统一模板

```text
实验ID：
配置文件：
基准实验ID：
唯一变化：
commit ID：
配置digest：
数据digest：
随机种子：
最佳epoch/最后epoch：
最佳Top-1/最后Top-1：
相对基准Top-1变化：
Macro及相对变化：
后25%类别准确率及相对变化：
高可信子集Top-1：
增强一致率：
与原始CLIP特征余弦：
训练-验证差：
trusted/uncertain/suspicious比例：
峰值显存和训练时间：
是否NaN/OOM/预测塌缩：
最佳checkpoint路径：
sample state路径：
你的初步判断：
```

另外增加各实验专属信息：

- B1：原型类内/类间距离、温度值。
- B2：LoRA层、rank、可训练参数数、合并等价误差。
- B3：信号分布、每类分区比例、是否有类别全部可疑。
- B4：ELR占总loss比例、最佳到最后的性能下降。
- B5：teacher梯度检查、feature cosine、额外显存时间。
- B6：weak/strong一致率、预测熵、强增强样例审计。

### 反馈频率

- 实验正常：结束后反馈一次完整摘要。
- 训练超过12小时：可在首个epoch和中程各发一次状态摘要。
- 触发异常：立即按 `F-ERROR` 反馈，不等待训练结束。

---

## 12. F09：冻结主线

### [本机或服务器] 你要做什么

汇总B0-B6，不能只选单次最高Top-1。确认每个保留模块都有明确收益。

### [反馈 F09] 发给我

```text
B0-B6汇总CSV路径：
各实验Top-1/Macro/困难类别指标：
各实验随机种子：
你希望保留的模块：
你希望淘汰的模块：
最高单次模型：
最稳定模型：
计算成本最低的候选：
是否存在规则疑问：
```

我会协助给出主线冻结结论和后续U系列优先级。

### [门禁]

没有统一验证集、配置digest或基准run的实验不能进入汇总。

---

## 13. F10：U系列上限方法

推荐顺序：

```text
U1 严格软伪标签
U2 课程学习
U3 多原型
U4 类别均衡（仅F05确认需要时）
U5 可信梯度约束
U6 自适应分区
```

每个U实验都从F09冻结主线出发，只增加一个模块。

### [反馈 F10] 除F08统一模板外，再发

```text
模块实际影响的样本数及比例：
受影响最多的类别：
头/中/困难类别变化：
标签或分区转移矩阵路径：
预测熵变化：
额外显存/训练时间：
关闭模块是否完全恢复主线：
是否增加随机波动：
```

U1伪标签额外反馈：覆盖率、每类流入/流出、最大类别占比、平均置信度、原标签冲突数。

### [门禁]

单种子无稳定提升的模块不进入组合实验。多个未经验证的U模块不得一起加入。

---

## 14. F11：多种子与第二划分复验

### [服务器] 你要做什么

1. 保持全部超参数不变，只更换预登记随机种子。
2. 至少复验两个种子；重要候选建议三个。
3. 必要时使用第二个固定训练/验证划分确认。
4. 计算均值、标准差和相对基准配对差值。

### [反馈 F11] 发给我

```text
候选方案：
每个种子的run ID：
每个种子的Top-1/Macro/困难类别：
均值和标准差：
相对同种子基准差值：
第二划分结果：
显存和时间均值：
是否每个种子方向一致：
是否存在某类严重退化：
```

我会依据 `DECISION_RULES.md` 给出 ACCEPT、CONDITIONAL 或 REJECT。

### [门禁]

只在一个种子上偶然提高的复杂模块不能进入最终模型。

---

## 15. F12：最终全数据训练

### [本机] 先冻结发布版本

```text
固定commit ID
固定容器image digest
固定最终resolved config
固定类别映射生成规则
```

### [服务器] 再训练

1. 使用全部官方训练数据。
2. 测试集仍不进入任何训练和阈值计算。
3. 重新计算全训练集原型和样本可信度。
4. 不复用旧验证划分产生的sample state。
5. 使用预先确定的训练轮数或停止规则。

### [反馈 F12] 发给我

```text
最终commit ID：
镜像名称和digest：
最终配置路径和digest：
全训练数据digest：
训练epoch/global step：
loss是否正常：
最终模型路径：
模型SHA256：
类别映射路径和digest：
训练总时间和峰值显存：
是否存在异常中断或恢复：
```

### [门禁]

最终模型来源、配置、数据digest或类别映射任何一个缺失时，先补齐追溯信息，不进入测试推理。

---

## 16. F13：单模型导出、测试推理与提交

### [服务器] 你要做什么

1. 合并LoRA。
2. 验证合并前后输出等价。
3. 确认导出包不包含teacher或第二模型。
4. 测试集只运行推理。
5. 输出原始四位类别ID。
6. 使用提交校验脚本检查CSV。
7. 保存CSV和压缩包SHA256。

### [反馈 F13] 提交前发给我

```text
最终模型文件：
是否只有一个推理模型：
LoRA合并等价最大误差：
推理图片数：
CSV行数：
是否有表头：
重复/缺失/多余文件数：
非法类别数：
CSV校验完整输出：
模型SHA256：
CSV SHA256：
压缩包SHA256：
```

不要发送完整测试数据或秘密凭证。

### [门禁]

CSV校验没有显示 `status: ok`，绝对不要提交。

---

## 17. F14：提交后与项目归档

### [反馈 F14] 发给我

```text
提交时间：
提交文件SHA256：
平台是否确认接收：
线上Top-1：
对应模型run ID：
对应commit ID：
是否出现格式或运行错误：
```

线上成绩只用于验证里程碑候选，避免根据排行榜高频调参。

服务器释放前必须备份：

- 最终模型；
- class mapping；
- resolved config；
- 数据manifest digest；
- 最佳/最终指标；
- 实验汇总表；
- 环境快照；
- 最终提交文件和哈希；
- 失败实验诊断记录。

---

## 18. F-ERROR：任何阶段的紧急反馈模板

出现以下情况立即停止：

- 测试集进入训练流程；
- 类别映射digest改变；
- 原始数据被修改；
- 未授权CLIP参数或teacher出现梯度；
- NaN/Inf；
- 预测塌缩到少数类别；
- 某类全部被判为可疑；
- checkpoint与sample state版本不一致；
- 磁盘将满；
- LoRA合并前后输出明显不一致；
- OOM重复发生。

立即反馈：

```text
当前阶段：
实验ID/run ID：
commit ID：
完整启动命令：
配置文件和digest：
错误发生时间/epoch/step：
完整错误栈：
错误前50-100行日志：
nvidia-smi输出：
磁盘剩余空间：
最后有效checkpoint：
manifest.json路径：
resolved_config.yaml路径：
是否可以稳定复现：
你已经做过哪些处理：
```

不要反复重启同一个run覆盖现场，也不要删除失败目录。修复后使用新实验ID。

---

## 19. 最简执行清单

如果不确定下一步，只检查下面哪一项尚未完成：

```text
[ ] F00 规则和数据格式确认
[ ] F01 Git与项目骨架
[ ] F02 模块逐批交付和测试
[ ] F03 首个可部署commit
[ ] F04 服务器健康检查
[ ] F05 数据审计
[ ] F06 S0冒烟测试
[ ] F07 B0基线
[ ] F08 B1-B6逐项消融
[ ] F09 主线冻结
[ ] F10 U系列单因素实验
[ ] F11 多种子复验
[ ] F12 最终全数据训练
[ ] F13 导出、推理、CSV校验和提交
[ ] F14 结果与产物归档
```

每完成一个 F 节点，就把该节点要求的信息反馈给主协作会话；确认门禁通过后再进入下一节点。
