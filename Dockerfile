# Base image with NVIDIA CUDA and cuDNN
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS builder

# Copy uv binaries from upstream image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies and create a non-root user
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        tini \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g 1001 whisper && \
    useradd -u 1001 -g whisper --create-home --shell /bin/bash whisper

# Switch to non-root user
USER whisper

# Model 및 HuggingFace 캐시 경로 고정
ENV TRANSFORMERS_CACHE=/home/whisper/.cache/huggingface
ENV WHISPER_CACHE=/home/whisper/.cache/whisper
ENV UV_COMPILE_BYTECODE=1

# Set working directory
WORKDIR /home/whisper

# Copy dependency files first to leverage build cache
COPY --chown=whisper:whisper pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-cache

# Copy the entire application code (including preload scripts)
COPY --chown=whisper:whisper app/ app/

# Run preload script to load all Whisper models (with cache)
RUN uv run app/preload/preload_all.py && \
    echo "Preload complete" && \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip ~/.cache/huggingface/datasets

# Expose the application port
EXPOSE 8484

# Use Tini as the init process to manage signal forwarding
ENTRYPOINT ["tini", "--"]

# Start the FastAPI server
CMD ["./.venv/bin/fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8484"]
