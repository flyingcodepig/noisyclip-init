FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       git \
       tini \
    && rm -rf /var/lib/apt/lists/*

COPY init_build/02_deployment/requirements.txt /tmp/noisyclip-requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip==25.1.1 \
    && python -m pip install --no-cache-dir -r /tmp/noisyclip-requirements.txt \
    && python -m pip check

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash app \
    && mkdir -p /workspace/project /data/train /data/test /runs /cache \
    && chown -R app:app /workspace /runs /cache

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/project/src \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg

WORKDIR /workspace/project
USER app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

