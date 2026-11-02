FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="AIC NoisyCLIP semifinal reproduction environment" \
      org.opencontainers.image.description="Runtime dependencies only; no project code, model weights, data, caches, or outputs" \
      org.opencontainers.image.source="https://github.com/flyingcodepig/noisyclip-init" \
      org.opencontainers.image.revision="88c69c00ca3a4b4715061eda2db9824a1d69e64c"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY requirements-runtime.txt /tmp/requirements-runtime.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && rm -f /tmp/requirements-runtime.txt \
    && rm -rf /root/.cache/pip /opt/conda/pkgs/* \
    && python -c "import torch, torchvision, transformers, numpy, PIL; assert torch.__version__ == '2.6.0+cu124'; assert torchvision.__version__ == '0.21.0+cu124'; assert transformers.__version__ == '4.46.3'; assert numpy.__version__ == '2.1.2'; assert PIL.__version__ == '11.0.0'; assert torch.version.cuda == '12.4'; print('environment versions verified')"

WORKDIR /workspace
CMD ["bash"]
