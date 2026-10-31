# 复赛 r0 基线交接文档

记录日期 2026-10-31。对象是 `release/rematch-r0-baseline` 分支上的复赛 r0 基线。本文件只写该分支里真实存在、且哈希已核对的内容；分支外的实验只做标注，不并入 r0 的成绩。

## 0. 三十秒版本

- **是什么**：复赛 r0 的单模型 EMA 权重（Git LFS）＋ 官方 CLIP ViT-B/32 初始化、148,695 张训练图的完整训练配置 ＋ 四视图读出与固定评分记录。
- **在哪**：产物在 `experiments/semifinal_a_doc/baseline_r0/`，代码在同级 `experiments/semifinal_a_doc/`，权重指针在 `experiments/semifinal_a_doc/checkpoints/r0_ema.pt`。
- **分数**：开发 20,767／26,218 = **79.2089%**，保留 8,823／11,226 = **78.5943%**，全体 29,590／37,444 = **79.0247%**。
- **不是什么**：不是队友 77.6293 分方案的逐位源码或权重复现；本分支不含训练／测试图像、测试标签、优化器断点、运行日志与各变体 CSV。

## 1. 分支与提交身份

- 仓库：<https://github.com/flyingcodepig/noisyclip-init>
- 分支：`release/rematch-r0-baseline`。r0 内容的树是 `d13a9bc8cbf12165a6bbb1f40b5cd7ec453896c4`（对应 `7aa6fbf`）；本交接文档是之后的提交，分支最新提交以 `git log --oneline` 为准，别把本文档的提交当成 r0 快照。
- 提交：
  - `a48ad0a` — 复赛 r0 基线代码、配置、四视图等权预测、开发／保留／全体评分
  - `7aa6fbf` — r0 最终 EMA 权重经 Git LFS 入库
  - `5b2332a` — 本交接文档与根 README 的实验记录入口（文档修订会再往后追加）
- 权重以 LFS 指针形式在树里（134 字节）：

  ```text
  version https://git-lfs.github.com/spec/v1
  oid sha256:e02506eb17e193c7b662a3d28fb96e9a6959f118c7e6d60569ac25b61a817f0e
  size 353468474
  ```

  克隆后需要 `git lfs install && git lfs pull`，会拉 353 MB。直接用 GitHub 的源码 ZIP 下载只会得到指针文件。

## 2. 分支上有什么

### 2.1 r0 产物：`experiments/semifinal_a_doc/baseline_r0/`

| 文件 | 内容 | 大小 | SHA256 |
|---|---|---:|---|
| `config.json` | r0 实际训练配置（与 `../config_rematch.json` 完全相同） | 1,170 | `377a501aa73849bad14d3432eb87557ba43dd2a1777e9f6262616e509c572444` |
| `equal.csv` | 37,444 行 `文件名,四位类别`，四视图概率等权 + E1 图传播的最终交付预测 | 1,572,648 | `c4b95a8b5c37776ca29996cafebe9ee45fcc9f78eeb372fd9914e5c8eb2a96d2` |
| `equal.score.json` | 开发集评分 | 175 | `3af62aa40c2a484c24bcabdfa7cbf195fc77ef4eae9171b456efff282032cba6` |
| `equal.holdout.json` | 保留集评分 | 170 | `a64876791a3c809488ad22dd7514af948ff21c90de9446a76e386c2d80fcdf1e` |
| `equal.full.json` | 全体 37,444 张评分 | 66 | `8d1f8fc5b0f43345f2664433cd9d7a0c4eb9516c505794a05852136a0a7b9cff` |
| `manifest.json` | 视点缓存与配置哈希、四个权重变体的 CSV 哈希与类别计数统计 | 1,515 | `c813faebb494beae2467699aab791e0e13e36e72790ccfba987aa6e3b2037b64` |
| `README.md` | 快照说明与关键哈希 | 1,307 | `d5cbe0455062db973f25146266bd29e7956fce852d0dee92cc297d4afebc17d2` |

`manifest.json` 里记录的 4 个视点权重变体（只有 `equal` 的完整 CSV 在本分支，其余只留哈希，便于以后核对）：

| 变体 | 四视图权重 | CSV SHA256 | 类别计数 CV | 类计数范围 |
|---|---|---|---|---|
| `documented` | 0.23 / 0.16 / 0.21 / 0.40 | `295e5ad81b58860c8be55fb851b70a8b8fb9f36e5720c8a25ccb1e6e11e06254` | 0.11288 | 19–72 |
| `equal` | 0.25 / 0.25 / 0.25 / 0.25 | `c4b95a8b5c37776ca29996cafebe9ee45fcc9f78eeb372fd9914e5c8eb2a96d2` | 0.11256 | 23–72 |
| `center_focus` | 0.15 / 0.15 / 0.25 / 0.45 | `9bc41220d03271b35f8117b07aed5629eb1d87c989b9be56ac727ae8477a9778` | 0.11161 | 20–72 |
| `full_focus` | 0.40 / 0.20 / 0.20 / 0.20 | `7d83713067ac6e20ebdbe4b38bb6eef970603f34db81295014ae8dc3e0a06792` | 0.11502 | 19–74 |

另记录：视点缓存 SHA256 `65fec97cffa0fea09bd8fd40c54f4273836d5900682163175359397d83fcf5c9`（`view_cache.pt`，不在分支）。

### 2.2 代码与入口：`experiments/semifinal_a_doc/`

| 路径 | 作用 |
|---|---|
| `prepare.py` | 从训练图提取冻结 CLIP 双视图教师特征，构造类中心、留一法一致性、候选可信集、canonical 图像分组 |
| `train.py` | 正式训练入口；`--stop-after` 用于烟雾检查，`--resume` 续训 |
| `deliver.py` | 固定测试推理；`--graph-mode e1|joint`，`--save-cache` / `--save-view-cache` / `--cache-only` 导出读出缓存 |
| `data.py`、`model.py` | 数据与视图增强、CLIP 骨架与分类头 |
| `validate_submission.py` | 提交 ZIP 结构与 CSV 顺序校验 |
| `fetch_weights.py` | 获取官方 CLIP 权重并核对 SHA256 |
| `score_rematch.py` | 固定类内切分评分（开发／`--holdout`），不参与训练 |
| `sweep_rematch_views.py` | 固定模型四视图权重测试，产出 `documented`／`equal`／`center_focus`／`full_focus` |
| `sweep_rematch_readout.py`、`sweep_rematch_final_balance.py` | 读出与最终均衡扫描（产物不在本分支） |
| `ptmap_rematch.py` | PT-MAP 风格适配试验（产物不在本分支，未晋级） |
| `config_rematch.json` | r0 配置副本（与 `baseline_r0/config.json` 相同） |
| `config_rematch_canonical.json`、`config_rematch_queue32.json` | r1 canonical 与队列容量变体配置快照（产物不在本分支） |
| `config.json` | 半决赛 A 路线配置（500 类、90,197 张），不是 r0 配置 |
| `run_rematch_baseline.sh`、`run_rematch_sweep.sh`、`run_rematch_canonical.sh`、`run_rematch_ptmap.sh` | 服务器端执行脚本；含 `/root/aic_a100` 与 `/mnt/复赛` 路径，需按实际环境修改 |
| `weights/clip-vit-base-patch32/` | 只有 `config.json`、`preprocessor_config.json`；`pytorch_model.bin` 被 `.gitignore` 排除，需用 `fetch_weights.py` 自行下载 |
| `docs/01_全部实验与算法路线总账.md`、`docs/02_A当前完整算法与最初方案对照.md` | 算法依据（全量实验台账与 A 路线对照） |
| `README_A100.md` | 半决赛 A 路线重实现在 A100 上的运行说明（4862 步那一次，不属于 r0） |
| `checkpoints/r0_ema.pt` | r0 最终 EMA 权重（LFS 指针） |
| `requirements.txt`、`.gitignore` | 依赖与忽略规则 |

## 3. r0 是什么

官方 OpenAI CLIP ViT-B/32 初始化（SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`），在复赛训练集上做单模型噪声标签微调：

- 数据：训练 148,695 张／750 类；固定测试 37,444 张。标签文件 SHA256 `360c8f1b106b8b4f97c9fcf1022ae8d7f16932337757f674d2ff4891f15d0d9c`（不在分支）。
- 规模：图像 448、中心裁剪 384；`batch 96 × accumulation 2` = 有效 batch 192；主训练 7,503 步 ＋ 尾段 512 步 = **8,015 步**；warmup 209；seed 41291。
- 优化：AdamW，学习率 head `7.2e-4`／pool `8.8e-4`／early `1.55e-5`／late `3.1e-5`，weight decay 0.048，梯度裁剪 1.0；EMA decay 0.999。
- 鲁棒损失：GCE `q=0.72`、混合 GCE `q=0.4`、CE 下限 0.025；mixup 概率 0.27（α 0.19）、cutmix 概率 0.27（α 0.76）；冻结教师权重 0.045、EMA 特征 0.02、SupCon 0.031（温度 0.1）、L2-SP 0.0068。
- 可信队列：每类 8、最大年龄 256 步、128 步后启用；软一致性 1,536 步开始、512 步爬升、最大间隔 2,048、权重 0.30。
- 读出：四视图 `full 1.0`、`center 1.0`、`center 0.86 + 水平翻转`、`center 0.74`；后验温度 2.85、头部先验权重 0.06（尺度 15.0）；E1 图 22 近邻、8 轮传播、α 0.984。
- 交付的 `equal.csv` 用四视图**等权 0.25**，不是配置里的文档原权重（0.23/0.16/0.21/0.40）。

## 4. 结果

| 方案 | 开发 26,218 | 保留 11,226 | 全体 37,444 |
|---|---:|---:|---:|
| 文档原权重（0.23/0.16/0.21/0.40） | 20,757 = 79.1708% | 8,818 = 78.5498% | 29,575 = 78.9847% |
| **四视图等权（本分支 `equal.csv`）** | **20,767 = 79.2089%** | **8,823 = 78.5943%** | **29,590 = 79.0247%** |

等权相对文档原权重：开发 +10 张、保留 +5 张、全体 +15 张，全体领先 0.0401 个百分点。单种子、无方差估计，**不能宣称稳定胜出**。

同期还试过（产物与评分均不在本分支，数字取自当时的实验记录，未在此复核）：最终学生权重开发 20,743；r1 canonical 历史共享全体 29,471；PT-MAP 风格 ε=0.1／0.25 开发 20,155／20,151，都不如上面两行。

## 5. 切分与评分

`score_rematch.py` 用固定类内哈希切分：每类按 `sha256("rematch-41291:" + 文件名)` 排序，前 70% 为开发集（26,218 张），后 30% 为保留集（11,226 张）。切分数量与队友总账一致，但**没有证据表明与队友历史切分文件逐张相同**，所以两边百分比不能当严格配对比较。

开发集被用于多轮方法选择，保留集也审查过候选，因此全体 79.0247% 不是"未使用过的最终测试分数"。评分只在预测冻结后进行，标签不进入训练、特征或推理公式。

## 6. 复现

数据和标签不随分支提供。以下命令按 `README_A100.md` 的方式准备目录，只是把角色对调为复赛数据：

```bash
export CLIP=weights/clip-vit-base-patch32          # fetch_weights.py 下载并核对
python prepare.py --train-root data/train --official-weights "$CLIP" \
    --output prepared --classes 750 --expected-train 148695     # 默认值 500/90197 是半决赛的，必须显式覆盖

python train.py --prepared prepared --official-weights "$CLIP" \
    --config config_rematch.json --run runs/smoke --workers 8 --stop-after 2   # 烟雾检查，不作正式起点
python train.py --prepared prepared --official-weights "$CLIP" \
    --config config_rematch.json --run runs/r0 --workers 8                     # 8,015 步，产出 runs/r0/final_ema.pt

python deliver.py --test-root data/test --official-weights "$CLIP" \
    --checkpoint runs/r0/final_ema.pt --output runs/r0/cache_only --cache-only \
    --save-view-cache runs/r0/view_cache.pt --workers 8
python sweep_rematch_views.py --cache runs/r0/view_cache.pt \
    --config runs/r0/config.json --output runs/r0/view_sweep

python score_rematch.py --labels labels.csv --prediction runs/r0/view_sweep/equal.csv
python score_rematch.py --labels labels.csv --prediction runs/r0/view_sweep/equal.csv --holdout
```

`run_rematch_baseline.sh`、`run_rematch_sweep.sh` 是服务器上实际用过的封装，路径写死在 `/root/aic_a100` 与 `/mnt/复赛`，换环境要改。

已知偏差：准备阶段的日志记录 `images=148695`；`prepare.py` 的候选可信集规则会新增若干候选，当时记录为 649 张，这份数字与队友文档的 560 不同，也是"不能称为逐位复现"的证据之一。

## 7. 校验清单

```bash
git lfs install && git lfs pull
git lfs ls-files                                  # 应列出 r0_ema.pt
sha256sum experiments/semifinal_a_doc/checkpoints/r0_ema.pt   # e02506eb… 353,468,474 字节
sha256sum experiments/semifinal_a_doc/baseline_r0/equal.csv   # c4b95a8b…
wc -l experiments/semifinal_a_doc/baseline_r0/equal.csv       # 37444
sha256sum experiments/semifinal_a_doc/baseline_r0/config.json # 377a501a…
```

`equal.csv` 的 37,444 行应覆盖全部 750 类，每类 23–72 张（见 `manifest.json` 的 `equal` 变体）。

## 8. 禁止与边界

- **不要**把本分支叫作队友 77.6293 分方案的复现；它只是按两份技术文档重实现后跑出的 r0。
- 复赛测试图只作固定推理；公布的复赛测试标签只在预测完成后用于评分。
- 半决赛那一次 4,862 步训练与 49,857 张提交属另一条线（`experiment/semifinal-a-doc-reimplementation-20261031`），其权重与预测不在本分支，且已按要求**不再训练或续训**。
- 等权相对文档权重的优势只有 0.04 个百分点，未做多种子验证。
- 本分支不存训练图、测试图、标签、`last.pt` 优化器断点、视点缓存与日志；需要它们请回服务器或本机实验目录。

## 9. 本分支之外，但同一实验线（不要混算）

- 半决赛首次重实现：4,862／4,862 步、500 类、49,857 张测试全部有有效预测，无官方分数。
- 完整 ASAM（复赛）：开发 20,810 = 79.3729%、保留 8,819、全体 29,629 = 79.1288%，全体比 r0 等权多 39 张，保留集少 4 张；单种子，未归档到本分支。
- r1 canonical、PT-MAP、读出与均衡扫描：均未超过 r0 的等权读出，结论见实验总账。

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-31 | 建立本分支，`a48ad0a` 提交 r0 代码、配置、预测与评分；`7aa6fbf` 提交 LFS 权重；本文件补写交接说明 |
