# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Colette from source
RUN git clone https://github.com/jolibrain/colette.git /app/colette && \
    cd /app/colette && pip install .

# Install tiktoken (used by Colette)
RUN pip install tiktoken

# Add Replicate wrapper files
COPY predict.py /app/predict.py
COPY replicate.yaml /app/replicate.yaml

WORKDIR /app
