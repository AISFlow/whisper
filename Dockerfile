# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ARG USERNAME=whisper
ARG USER_UID=1001
ARG USER_GID=1001

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME_DIR=/home/${USERNAME} \
    VENV_PATH=/home/${USERNAME}/.venv/
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ffmpeg tini curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -u ${USER_UID} -g ${USERNAME} --create-home --shell /bin/bash ${USERNAME}

COPY --link --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# =====================================================================
FROM base AS builder

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME_DIR}
RUN chown -R ${USERNAME}:${USERNAME} ${HOME_DIR}
USER ${USERNAME}

RUN uv venv --python 3.12 --seed
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock ./
RUN uv sync --frozen
RUN ls -al

COPY --chown=${USERNAME}:${USERNAME} . .
RUN \
    echo "--- Preloading models... ---" && \
    uv run app/preload/preload_all.py --engine faster && \
    uv run app/preload/preload_all.py --engine openai && \
    echo "--- Preload complete. Cleaning up caches. ---" && \
    rm -rf .cache/huggingface/datasets

# =====================================================================
FROM base AS final

WORKDIR ${HOME_DIR}

COPY --link --from=builder --chown=${USERNAME}:${USERNAME} ${VENV_PATH} ${VENV_PATH}
COPY --link --from=builder --chown=${USERNAME}:${USERNAME} ${HOME_DIR}/.cache/ .cache/
COPY --link --from=builder --chown=${USERNAME}:${USERNAME} ${HOME_DIR}/app/ app/

USER ${USERNAME}
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/ || exit 1

ENTRYPOINT [ "tini", "--", "/opt/nvidia/nvidia_entrypoint.sh" ]

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]