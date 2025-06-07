FROM ghcr.io/pytorch/pytorch-nightly:2.3.0.dev20240305-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV PIP_CACHE_DIR=/root/.cache/pip

COPY requirements.txt /workspace/
RUN pip install \
    --cache-dir ${PIP_CACHE_DIR} \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY . /workspace
RUN pip install -e ./model/dynamic-network-architectures-main
