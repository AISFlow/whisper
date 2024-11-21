FROM python:3.12-bookworm AS builder

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U openai-whisper && \
    find /usr/local/bin/python -type f \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' \) -exec bash -c 'echo "Deleting {}"; rm -f {}' \;
