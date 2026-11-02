# 半决赛复现环境镜像

本目录只用于构建运行环境。镜像不包含队伍源码、`final_ema.pt`、官方
CLIP ViT-B/32 权重、赛事数据、缓存或预测结果。

环境基线来自正式 release commit：
`88c69c00ca3a4b4715061eda2db9824a1d69e64c`。

## 本地构建

```bash
cd docker_repro_env
docker build --pull \
  --tag aic-noisyclip-env:semifinal-cu124-20261102 \
  .
```

`.dockerignore` 默认排除全部文件，只允许 Dockerfile 和运行时依赖清单进入
构建上下文。

## 环境检查

```bash
docker run --rm --gpus all --network none \
  aic-noisyclip-env:semifinal-cu124-20261102 \
  python -c "import torch,torchvision,transformers,numpy,PIL; print(torch.__version__, torchvision.__version__, transformers.__version__, numpy.__version__, PIL.__version__); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
```

关键版本：Python 3.11、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、
Transformers 4.46.3、NumPy 2.1.2、Pillow 11.0.0。

## 宿主机挂载与推理

```bash
docker run --rm --gpus all --network none --shm-size=16g \
  --mount type=bind,src=/HOST/CODE,dst=/workspace/code,readonly \
  --mount type=bind,src=/HOST/FINAL_EMA,dst=/workspace/models/final_ema.pt,readonly \
  --mount type=bind,src=/HOST/OFFICIAL_CLIP,dst=/workspace/models/official-clip,readonly \
  --mount type=bind,src=/HOST/TEST,dst=/workspace/data/test,readonly \
  --mount type=bind,src=/HOST/CACHE,dst=/workspace/cache \
  --mount type=bind,src=/HOST/OUTPUT,dst=/workspace/output \
  --env AIC_CHECKPOINT=/workspace/models/final_ema.pt \
  --env AIC_CACHE_DIR=/workspace/cache \
  aic-noisyclip-env:semifinal-cu124-20261102 \
  bash /workspace/code/run_predict.sh \
    /workspace/data/test /workspace/models/official-clip /workspace/output
```

代码、最终权重、官方 CLIP 权重和测试数据均为宿主机只读挂载；缓存与输出目录
可写。完整推理需要 49,857 张半决赛 TEST 图片和完整的官方 CLIP ViT-B/32
Hugging Face 兼容权重目录。

预期输出为 `/workspace/output/pred_results.csv` 和 `submission.zip`。CSV 应有
49,857 行且无表头。

## GitHub Actions 输出

工作流会把镜像推送到 GHCR，并把 `docker save` 生成的 gzip 文件切分后作为
Actions artifact 上传。下载全部分片后合并：

```bash
cat aic-noisyclip-env_semifinal-cu124-20261102.tar.gz.part-* \
  > aic-noisyclip-env_semifinal-cu124-20261102.tar.gz
sha256sum -c aic-noisyclip-env_semifinal-cu124-20261102.tar.gz.sha256
docker load < aic-noisyclip-env_semifinal-cu124-20261102.tar.gz
```
