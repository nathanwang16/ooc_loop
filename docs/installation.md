# Installation

## Prerequisites

| Component  | Required version | Notes |
|------------|------------------|-------|
| Python     | 3.10, 3.11, 3.12 | `3.11` is the CI default |
| OpenFOAM   | v2406 (OpenFOAM.com / ESI-OpenFOAM) | macOS: install via `brew install gerlero/openfoam/openfoam@2406`; see `tip.md` |
| git        | any recent       |       |

CadQuery pulls in OCCT through `pip`; no system-level OCCT install is
required for the Python side.

## Option A — Docker (recommended for first use)

The Docker image bundles OpenFOAM 2406 + Python 3.11 + the package already
installed in editable mode.  Start an interactive shell with the repository
mounted at `/workspace`:

```bash
git clone https://github.com/ooc-loop/tumor-chip-design.git
cd tumor-chip-design

docker compose -f docker/docker-compose.yml run --rm tumor-chip-shell
# inside the container:
tumor-chip version
tumor-chip verify-scalar
```

All CFD runs produce artefacts under `./data/` on the host, so results
survive container restarts.

## Option B — Local install (macOS / Linux)

1. Install OpenFOAM 2406.  On macOS:

   ```bash
   brew tap gerlero/openfoam
   brew install gerlero/openfoam/openfoam@2406
   ```

   Commands are then accessed via the `openfoam2406` wrapper
   (e.g. `openfoam2406 -c 'simpleFoam -case ...'`).  The package's
   subprocess helpers auto-detect this wrapper; see
   `ooc_optimizer/cfd/scalar.py::_find_openfoam_prefix`.

2. Create a dedicated Python environment (conda `ooc` is the convention
   used throughout the project):

   ```bash
   conda create -n ooc python=3.11
   conda activate ooc
   ```

3. Install `tumor-chip-design`:

   ```bash
   git clone https://github.com/ooc-loop/tumor-chip-design.git
   cd tumor-chip-design
   pip install -e ".[dev,test,docs]"
   pre-commit install  # optional but strongly recommended for contributors
   ```

4. Verify the install:

   ```bash
   tumor-chip version
   pytest -q --deselect tests/test_verification.py::TestOpenFOAMIntegration
   ```

## Option C — PyPI (users only)

If you do not plan to modify the code:

```bash
pip install tumor-chip-design
```

After install you still need a working OpenFOAM 2406 on `PATH` (or the
`openfoam2406` wrapper) for anything that actually runs CFD.

## Troubleshooting

- **`cadquery` install fails on Linux**: install OCCT system libraries
  (e.g. `apt-get install libgl1`).  The Docker image already does this.
- **`codedFixedValue` fails on macOS DMG-mounted OpenFOAM**: see `tip.md`.
- **`torch` install slow on macOS**: use the default CPU wheel from PyPI
  (`pip install "torch>=2.1"`); no GPU acceleration is needed.
