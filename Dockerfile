FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

# Set environment variables
ENV TORCH_HOME=/workspace/models

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg tini && \
    rm -rf /var/lib/apt/lists/*

# Create a group and user named 'whisper'
RUN groupadd -g 1000 whisper && \
    useradd -u 1000 -g whisper -d /workspace -s /bin/bash whisper
# Create /workspace directory and set ownership
RUN mkdir -p /workspace/output && mkdir -p ${TORCH_HOME} && \
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
        ffmpeg-python

# Pre-load the 'turbo' model to cache it (optional)
RUN python3 -c "import whisper; whisper.load_model('turbo')"

# Copy application code
COPY --chown=whisper:whisper ./main.py /workspace/main.py

# Expose the port
EXPOSE 8484

ENTRYPOINT ["tini", "--"]

# Start the Uvicorn server with the application
CMD ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8484"]
