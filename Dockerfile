FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

# Set environment variables
ENV TORCH_HOME=/workspace/models \
    PYTHONUNBUFFERED=1 \
    PATH="/workspace/.local/bin:${PATH}"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg tini && \
    rm -rf /var/lib/apt/lists/* && \
        groupadd -g 1000 whisper && \
        useradd -u 1000 -g whisper -d /workspace -s /bin/bash whisper && \
    mkdir -p /workspace/output && mkdir -p ${TORCH_HOME} && \
        chown -R whisper:whisper /workspace

# Switch to the 'whisper' user
USER whisper

# Set working directory
WORKDIR /workspace

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir -U \
        openai-whisper \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        ffmpeg-python && \
    python3 -c "import whisper; whisper.load_model('turbo')"

# Copy application code
COPY --chown=whisper:whisper ./main.py /workspace/main.py

# Expose the port
EXPOSE 8484

ENTRYPOINT ["tini", "--"]

# Start the Uvicorn server with the application
CMD ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8484"]
