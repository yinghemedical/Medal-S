FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime  # 或 python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN cd model
RUN pip install -e dynamic-network-architectures-main

CMD ["python", "inference_cvpr25.py"]
