# syntax=docker/dockerfile:1.7
#
# tumor-chip-design runtime image.
#
# Base: OpenFOAM v2406 (OpenCFD / ESI-OpenFOAM "com" fork) — matches the
# development tip.md guidance and the verification test matrix in
# tests/test_verification.py.
#
# We layer Python 3.11 on top, install the package in editable mode via the
# [test,docs] extras, and expose `tumor-chip` as the default entry point.
#

FROM opencfd/openfoam-default:2406

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --- system deps ------------------------------------------------------------
# Python 3.11 from deadsnakes, plus libraries pulled in by cadquery / cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        ca-certificates \
        curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- workspace --------------------------------------------------------------
WORKDIR /workspace

# Install the package first using only the manifest so docker build caches
# the dependency layer.
COPY pyproject.toml README.md LICENSE CITATION.cff CHANGELOG.md ./
COPY ooc_optimizer ./ooc_optimizer
COPY tumor_chip_design ./tumor_chip_design
COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
 && python -m pip install -e ".[test,docs]"

# --- remainder of the repo --------------------------------------------------
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples
COPY paper ./paper
COPY Development_Guide_v2.md ./

# OpenFOAM env is sourced by the upstream base image's entrypoint; the
# tumor-chip CLI wraps Python subprocesses that inherit the parent env.
ENTRYPOINT ["tumor-chip"]
CMD ["--help"]
