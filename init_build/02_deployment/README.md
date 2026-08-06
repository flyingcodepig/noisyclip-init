# 服务器选择与部署方案

## 1. 推荐服务器

### 主推荐：单卡 48GB

```text
GPU: 1 x NVIDIA L40S 48GB / RTX 6000 Ada 48GB / RTX A6000 48GB
CPU: 24-32 vCPU
RAM: 128GB
系统盘: 100GB SSD
数据与实验盘: 2TB NVMe，持续读写建议 >= 1GB/s
网络: 首次下载镜像与官方 CLIP 权重时需要稳定公网；训练可离线
OS: Ubuntu 24.04 LTS x86_64
```

这是主路线最省心的配置。48GB 能同时容纳 student、冻结 teacher、weak/strong 双视图、样本状态和较大的 batch，减少梯度累积以及 OOM 调参次数。

### 预算方案：单卡 24GB

```text
GPU: 1 x RTX 4090/4090D 24GB
CPU: 16 vCPU
RAM: 64GB
实验盘: 1TB NVMe
```

适合 B0-B6 主线，但建议 batch 32-64、AMP、梯度累积，并按需离线缓存冻结 teacher 特征。不要在同一 GPU 并行启动两个训练作业。

### 高吞吐方案

```text
GPU: 2 x RTX 4090 24GB，或 1 x A100 80GB
CPU: 32 vCPU
RAM: 128-256GB
实验盘: 2TB NVMe
```

双卡优先用于同时跑两个独立单模型消融，而不是为了最终多模型集成。每个 run 必须独占一张 GPU 和独立输出目录。

### 不推荐

- T4/P100：吞吐和显存限制会显著拖慢迭代。
- 16GB 及以下 GPU：双视图、teacher 和 LoRA 联合训练容易频繁 OOM。
- 机械硬盘或网络盘直接读取 10 万张小图片：DataLoader 容易成为瓶颈。
- H100：可以使用，但对 ViT-B/32 初赛主线通常性价比不高。

租赁时优先比较：显存是否独享、NVMe 是否本地盘、磁盘 IOPS、是否可制作快照、GPU 驱动版本、出站流量限制，而不只看每小时价格。

## 2. 固定软件基线

```text
Host OS: Ubuntu 24.04 LTS
NVIDIA driver: 570 或更高
Docker Engine: 官方 apt stable
NVIDIA Container Toolkit: 1.19.1-1
Container: pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
Python: 镜像内版本
PyTorch: 2.11.0
torchvision: 与镜像配套版本
```

选择 CUDA 12.8 而不是追逐最新 CUDA 13，是为了兼顾 Ada/Ampere GPU、成熟生态和可复现性。宿主机只安装 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；Python/CUDA 用户态依赖放入容器。

官方依据：

- [PyTorch 官方镜像标签](https://hub.docker.com/r/pytorch/pytorch/tags/)
- [PyTorch 历史版本安装矩阵](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.8 驱动兼容说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [Docker Engine Ubuntu 安装文档](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit 安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 3. 服务器目录布局

```text
/opt/noisyclip/project/             # Git 项目和 init_build
/mnt/noisyclip-data/init/train/     # 官方训练数据，只读挂载
/mnt/noisyclip-data/init/test/      # 官方测试数据，只读挂载
/mnt/noisyclip-runs/                # 实验输出、checkpoint
/mnt/noisyclip-cache/               # 官方 CLIP 权重和包缓存
/mnt/noisyclip-backup/              # 可选的异盘/对象存储同步目录
```

禁止把原始数据放在项目目录中，禁止让训练容器对原始数据拥有写权限。

空间预算建议：

- 原始数据：按实际大小的 1.2 倍预留。
- 特征缓存：`N x 512 x 4 bytes` 约 0.2GB，但还需索引和多版本原型。
- 单个完整 checkpoint：预留 1-3GB。
- 20-40 个实验及日志：至少 300GB。
- Docker 镜像、构建缓存和官方权重：至少 50GB。

## 4. 首次部署步骤

### 4.1 选择带 NVIDIA 驱动的云镜像

先执行：

```bash
nvidia-smi
```

若命令不存在或驱动低于 570，优先在云平台控制台更换官方 GPU 镜像。不要在长时间训练前临时升级驱动，因为驱动升级通常需要重启。

### 4.2 安装 Docker 和 NVIDIA 容器运行时

```bash
cd /opt/noisyclip/project/init_build/02_deployment
bash bootstrap_ubuntu_24_04.sh --yes
```

脚本不会安装或替换 GPU 驱动；它会在驱动不满足条件时终止。

### 4.3 配置路径

```bash
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，所有路径使用绝对路径。确认训练和测试目录与项目/输出目录不同。

### 4.4 构建并健康检查

```bash
bash deploy.sh --env .env --build
bash deploy.sh --env .env --check
```

健康检查必须确认：GPU 可见、CUDA 可用、PyTorch 版本正确、GPU 显存满足最低值、四个挂载目录权限正确。

### 4.5 进入容器

```bash
docker compose --env-file .env -f compose.yaml run --rm train bash
```

进入容器后先运行数据审计和两 batch smoke test，不要立即启动完整训练。

## 5. 日常运行

训练由 `04_scripts_and_configs/scripts/run_experiment.sh` 启动。推荐使用 `tmux` 或云平台作业调度器；不要仅依赖 SSH 会话。

```bash
tmux new -s noisyclip
cd /workspace/project
bash init_build/04_scripts_and_configs/scripts/run_experiment.sh \
  init_build/04_scripts_and_configs/configs/experiments/b0_frozen_linear.yaml \
  0
```

每次训练前运行 preflight；每次训练后检查 `DONE`、最佳 checkpoint、环境快照和指标文件。服务器关机前，把重要 run 同步到独立存储。

## 6. 安全与运维

- SSH 只开放密钥登录；不要将 TensorBoard 暴露到公网。
- 需要查看 TensorBoard 时使用 SSH 隧道。
- `.env`、云密钥和对象存储密钥不得提交 Git。
- 原始训练/测试数据不得上传到公共仓库或与比赛无关的位置。
- 租赁实例删除前确认 checkpoint、resolved config、class mapping 和指标已备份。
- 容器镜像和依赖锁定后，不要在实验中途执行无版本约束的 `pip install -U`。

