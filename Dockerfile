FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TZ="Asia/Seoul" \
    USER=whisper \
    UID=1001 \
    GID=1001 \
    GOSU_VERSION=1.17 \
    TINI_VERSION=v0.19.0

ENV PATH="/home/${USER}/.venv/bin:/home/${USER}/.local/bin:$PATH"

RUN \
    --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update -qq; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        ffmpeg \
        curl \
        wget \
        build-essential \
        git \
        ca-certificates \
        gnupg; \
    \
    dpkgArch="$(dpkg --print-architecture)"; \
    wget -O /usr/bin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${dpkgArch}"; \
    wget -O /usr/bin/tini.asc "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${dpkgArch}.asc"; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys 595E85A6B1B4779EA4DAAEC70B588DFF0527A9B7; \
    gpg --batch --verify /usr/bin/tini.asc /usr/bin/tini; \
    rm -rf "$GNUPGHOME" /usr/bin/tini.asc; \
    chmod +x /usr/bin/tini; \
    tini --version; \
    \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${dpkgArch}"; \
    wget -O /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${dpkgArch}.asc"; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4; \
    gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu; \
    rm -rf "$GNUPGHOME" /usr/local/bin/gosu.asc; \
    chmod +x /usr/local/bin/gosu; \
    gosu --version; \
    gosu nobody true; \
    \
    groupadd --gid ${GID} ${USER}; \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USER}; \
    \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*;

USER ${USER}
WORKDIR /home/${USER}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv python install 3.12.9 --default --preview && \
    uv tool update-shell && \
    uv venv --python 3.12.9 --seed 

# ────────────── builder ──────────────
FROM base AS builder
# RUN python --version && \
#     which python && \
#     python -c "import sys; print(sys.executable, sys.version, sys.platform)"

COPY --link --chown=${UID}:${GID} pyproject.toml uv.lock ./
COPY --link --chown=${UID}:${GID} app/ app/

RUN uv sync --frozen && \
    uv run python app/preload/preload_all.py \
        --engine faster \
        --model large-v3 &&\
    uv cache clean

RUN grep -q -F 'source "/home/${USER}/.venv/bin/activate"' /home/${USER}/.bashrc || \
    echo 'if [ -f "/home/${USER}/.venv/bin/activate" ]; then source "/home/${USER}/.venv/bin/activate"; fi' >> /home/${USER}/.bashrc

# ────────────── final ──────────────
FROM base AS final

COPY --link --from=builder --chown=${UID}:${GID} /home/${USER} /home/${USER}
COPY --link --from=builder --chown=${UID}:${GID} /home/${USER}/.cache/huggingface /home/${USER}/.cache/huggingface
COPY --link --chown=${UID}:${GID} endeavour /usr/bin/endeavour
EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
#     CMD curl --fail http://localhost:8000/ || exit 1
USER root

ENTRYPOINT ["tini", "--", "/opt/nvidia/nvidia_entrypoint.sh"]
CMD [ "endeavour" ]
