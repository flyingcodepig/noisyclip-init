# 复赛 r0 基线快照

本目录记录 2026-10-31 的复赛 r0：官方 CLIP ViT-B/32 初始化，148,695 张复赛训练图、750 类，有效 batch 192，7503 主步＋512 尾步。训练入口为相邻目录的 `prepare.py`、`train.py`，推理入口为 `deliver.py`、`sweep_rematch_views.py`。`config.json` 是实际训练配置；`../checkpoints/r0_ema.pt` 是最终 EMA 单模型权重，由 Git LFS 存储。

`equal.csv` 为该权重对固定 37,444 张复赛测试图的四视图等权 E1 预测。开发集 20767/26218，保留集 8823/11226，全体 29590/37444。切分由 `score_rematch.py` 计算。本次记录不代表队友历史 77.6293 分实现的逐位源码或权重。

| 文件 | SHA256 |
|---|---|
| `config.json` | `377a501aa73849bad14d3432eb87557ba43dd2a1777e9f6262616e509c572444` |
| `../checkpoints/r0_ema.pt` | `e02506eb17e193c7b662a3d28fb96e9a6959f118c7e6d60569ac25b61a817f0e` |
| `equal.csv` | `c4b95a8b5c37776ca29996cafebe9ee45fcc9f78eeb372fd9914e5c8eb2a96d2` |
| 官方 CLIP 初始化 | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |

训练图、测试图、公布的测试标签、训练缓存和优化器恢复断点不在此分支。官方 CLIP 初始化由 `fetch_weights.py` 获取。复赛测试标签仅用于预测完成后的评分。
