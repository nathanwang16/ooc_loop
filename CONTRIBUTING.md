# Contributing to tumor-chip-design

Thanks for taking the time to contribute.  This document explains how to
report problems, propose changes, and work on the code.

## Scope of the project

`tumor-chip-design` is a research codebase paired with a Lab-on-a-Chip
manuscript and a JOSS companion paper.  The primary scientific contribution
is the interpretability output (Module 3.3 — Sobol indices, GP gradients,
tolerance intervals).  Contributions that improve the interpretability layer
or add new inverse-design target classes are especially welcome.

The framework is also intentionally generalisable: `examples/wss_uniformity/`
demonstrates the same engine solving a completely different scalar field
(wall shear stress instead of tracer concentration).  Contributions that
demonstrate other scalar problems (e.g. oxygen transport, temperature
uniformity) are welcome as new `examples/...` directories.

## How to report a problem

Open a [GitHub issue](https://github.com/ooc-loop/tumor-chip-design/issues)
using one of the two templates:

- **Bug report** — please include the full config YAML, the command you ran,
  the exact Python / OpenFOAM versions, and the traceback (or the contents
  of `simpleFoam.log` / `scalarTransportFoam.log` when CFD is involved).
- **Feature request** — describe the research question first, then the
  smallest API change that would unblock it.

## How to propose a change

1. Fork the repository and create a topic branch from `main`.
2. Install the development environment:
   ```bash
   pip install -e ".[dev,test,docs]"
   pre-commit install
   ```
3. Make your change in small, coherent commits.
4. Run the pre-submission checks locally:
   ```bash
   ruff format .
   ruff check .
   mypy ooc_optimizer
   pytest
   ```
   CI will run the same checks on Python 3.10, 3.11, and 3.12.
5. Update `CHANGELOG.md` under the `[Unreleased]` section.
6. Open a PR against `main` and fill in the PR template.

## Coding conventions

- **Style**: `ruff format` (double quotes, 100-column limit).
- **Linting**: `ruff check` selects `E, F, W, I, UP, B, C4, SIM, RET, PT,
  NPY`; see `pyproject.toml` for the exact rules and exemptions.
- **Typing**: `mypy --strict` on `ooc_optimizer/` (tests are permissive).
- **Docstrings**: NumPy style.  The public API in
  `ooc_optimizer/{cfd,geometry,optimization,interpretability,validation}/`
  is auto-documented via `mkdocstrings`, so docstrings are user-facing.
- **No placeholders**: per the project rule in `CLAUDE.md`, do not ship code
  with default stubs or "TODO: implement" bodies.  Raise `ValueError` or
  `NotImplementedError` with a clear message when a path is legitimately
  incomplete.
- **No script-level file writes in tests**: tests use `tmp_path`.
- **Do not commit large binary artefacts**: keep `.stl`, `.png`, and
  OpenFOAM case files out of `git` — the `data/` directory is already in
  `.gitignore`.

## Testing conventions

Tests live in `tests/` and are split into three tiers:

| Tier        | Runtime    | Marker                  | When it runs |
|-------------|------------|-------------------------|--------------|
| Unit        | < 1 s each | (default, no marker)    | Every CI job |
| Integration | < 30 s     | `@pytest.mark.integration` | Every CI job |
| Slow / OpenFOAM | > 1 min | `@pytest.mark.slow` / `@pytest.mark.openfoam` | Nightly only |

Unit tests **must not** call out to the real OpenFOAM solver.  Mock the
subprocess or write dictionary-file tests instead.  Integration tests may
run a small mesh + solve (see `tests/test_verification.py::TestOpenFOAMIntegration`).

## Reviewing and merging

- A PR requires one approving review from a maintainer and green CI.
- Squash-merge with a Conventional-Commits message (`feat:`, `fix:`,
  `docs:`, `refactor:`, `test:`, `chore:`).
- Tag `v0.X.Y` on `main` to trigger the release workflow.

## Releases

The release workflow (`.github/workflows/release.yml`) is triggered by a
tag of the form `vX.Y.Z`.  It builds wheels, uploads them to PyPI, and
archives the release to Zenodo (which assigns a DOI shown as a README
badge).  Only maintainers can tag releases.

## Code of conduct

All contributors are expected to follow the
[Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
