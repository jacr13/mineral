# --------- base ---------
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base

# Build-time UID/GID (defaults let you build without args)
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create non-root user
RUN groupadd -g ${GROUP_ID} noroot && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash noroot

# System deps
RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential g++ cmake ninja-build git curl wget \
    xvfb unzip patchelf ffmpeg swig graphviz \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Work as the non-root user from here on
USER ${USER_ID}:${GROUP_ID}
ENV HOME=/home/noroot
WORKDIR ${HOME}

# --------- Miniconda (in user's HOME) ---------
FROM base AS conda_setup

ENV PATH=${HOME}/conda/bin:$PATH
RUN set -eux; \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.4.0-0-Linux-x86_64.sh"; \
    wget -q "${MINICONDA_URL}" -O miniconda.sh; \
    bash miniconda.sh -b -p "${HOME}/conda"; \
    rm miniconda.sh; \
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> "${HOME}/.bashrc"; \
    echo "conda activate base" >> "${HOME}/.bashrc"

# --------- Poetry (user install) ---------
FROM conda_setup AS poetry_setup

RUN pip install --no-cache-dir poetry && \
    poetry self add poetry-dotenv-plugin && \
    poetry config virtualenvs.create false

WORKDIR /workspace
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml /workspace/
RUN poetry install -vvv --no-interaction --no-root

# --------- Final image ---------
FROM poetry_setup AS final
# Writable workspace + git safety (useful when bind-mounting)
RUN mkdir -p /workspace && chmod -R a+rwX /workspace && \
    git config --global --add safe.directory /workspace

# Default shell
CMD ["/bin/bash"]
