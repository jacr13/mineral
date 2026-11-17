# syntax=docker/dockerfile:1

ARG UV_VERSION=0.13.7
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# The development image supplies nvcc for DFlex's CUDA kernel compilation.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    build-essential cmake ninja-build git curl wget \
    xvfb xauth unzip patchelf ffmpeg swig graphviz \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create writable directories before switching away from root.
RUN groupadd -g "${GROUP_ID}" noroot \
 && useradd -m -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash noroot \
 && install -d -o "${USER_ID}" -g "${GROUP_ID}" \
    /workspace /opt/python /opt/venv /opt/nablaengine /home/noroot/.cache \
 && install -d -m 1777 /tmp/.X11-unix

ENV HOME=/home/noroot \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_PROGRESS=1

COPY --from=uv /uv /uvx /usr/local/bin/

USER ${USER_ID}:${GROUP_ID}
WORKDIR ${HOME}

# uv installs Python without a bootstrap Python or Conda environment.
FROM base AS python_setup
ARG PYTHON_VERSION=3.12.16
RUN uv python install --no-bin "${PYTHON_VERSION}"

# Keep the project environment outside the bind-mounted source directory.
FROM python_setup AS dependencies
WORKDIR /workspace
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml uv.lock /workspace/
RUN --mount=type=cache,id=mineral-uv-${USER_ID}-${GROUP_ID},target=/home/noroot/.cache/uv,uid=${USER_ID},gid=${GROUP_ID},sharing=locked \
    uv sync --locked --no-install-project --python "${PYTHON_VERSION}"

FROM dependencies AS final
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN git config --global --add safe.directory /workspace

# Match DFlex's portable PTX target without detecting GPUs on the build host.
# Limit parallel compilation to keep memory use manageable.
ARG TORCH_CUDA_ARCH_LIST="5.0+PTX"
ARG MAX_JOBS=2
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

# The pinned DFlex revision passes -Z to GCC, which does not support it.
adjoint = Path(find_spec("dflex").origin).with_name("adjoint.py")
adjoint.write_text(adjoint.read_text().replace(
    'cpp_flags = ["-Z", "-O2", "-DNDEBUG"]',
    'cpp_flags = ["-O2", "-DNDEBUG"]',
))

# DFlex gates CUDA compilation on device availability, although nvcc needs no GPU.
# Override the check only in this build process; runtime detection stays normal.
with patch("torch.cuda.is_available", return_value=True):
    import dflex
PY

# NablaEngine (G1 motion tracking, ICML plan X9): the package comes from uv.lock; its G1 model and reference data live
# in the repository's fixtures/ (and test states in benchmarks/), so keep a checkout of the same commit.
ARG NABLAENGINE_REV=31842fb
RUN git clone -q https://github.com/sailab-ai/NablaEngine.git /opt/nablaengine \
 && git -C /opt/nablaengine checkout -q "${NABLAENGINE_REV}"
ENV DK_FIXTURES=/opt/nablaengine/fixtures/mjlab \
    DK_BENCH_DATA=/opt/nablaengine/benchmarks/data \
    NVCC=/usr/local/cuda/bin/nvcc \
    DK_CACHE_DIR=/opt/nablaengine_cache
# Prebuild the engine's PyTorch library for the cluster's GPU architectures (no GPU needed at build time): Turing
# (sm_75), A100 (sm_80), RTX 3090 / A5000 / A6000 (sm_86), Ada (sm_89), H100 (sm_90). At run time init() only loads
# it (the image is read-only). A GPU outside this list: point DK_CACHE_DIR to a writable folder and it is built there.
USER root
RUN install -d -o 1000 -g 1000 /opt/nablaengine_cache
USER 1000:1000
RUN python -m nablaengine.torch_ffi build --arch sm_75 --arch sm_80 --arch sm_86 --arch sm_89 --arch sm_90 \
 && ls /opt/nablaengine_cache

CMD ["/bin/bash"]
