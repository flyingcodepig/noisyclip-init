# 数据放置说明

使用本阶段官方500类数据。相对于代码目录，训练图像放在`data/semifinal/train/0000/`至`data/semifinal/train/0499/`，完整90,197张；测试图像直接放在`data/semifinal/test/`，完整49,857张。保持原始文件名和图像字节。

原图由复现者取得。推理只需要TEST原图与包内模型；重训需要TRAIN原图、随包未训练官方特征和下载的固定官方初始化。`inputs/train_manifest.jsonl`记录TRAIN原图顺序、观察标签及SHA256，`inputs/test_names.json`记录完整TEST文件名顺序。

已有数据无需复制：预测用`--test-dir`指定，缓存重建用`--train-dir`指定。数据损坏、缺失或重复时程序终止。完整缓存约66.1GiB，由`ops/cache_train.py`重建并核对哈希。
