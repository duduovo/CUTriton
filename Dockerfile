FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04@sha256:cfa531cab5e50ca43b1ca73519586c7da7b797f8574a8fce8b63ca34e86e63bd

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUTRITON_MODEL=/models/model.onnx \
    CUTRITON_CACHE=/var/cache/cutriton \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 cutriton \
    && useradd --uid 10001 --gid cutriton --create-home --shell /usr/sbin/nologin cutriton

WORKDIR /opt/cutriton
COPY requirements.lock pyproject.toml README.md LICENSE ./
COPY python ./python
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip==25.2 \
    && /opt/venv/bin/pip install -r requirements.lock \
    && /opt/venv/bin/pip install --no-build-isolation --no-deps . \
    && mkdir -p /models /var/cache/cutriton \
    && chown -R cutriton:cutriton /models /var/cache/cutriton

USER 10001:10001
EXPOSE 8001 8002
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
  CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/ready', timeout=2)"]
ENTRYPOINT ["/opt/venv/bin/cutriton-serve"]
CMD ["/models/model.onnx", "--cache-dir", "/var/cache/cutriton"]
