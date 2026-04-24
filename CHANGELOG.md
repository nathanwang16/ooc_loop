# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- `v0.7.0` — migrate the implementation from `ooc_optimizer/` onto the
  `src/tumor_chip_design/` layout described in Development Guide v2 §5.1.
  The public `tumor_chip_design` namespace has been stable since `v0.6.0`,
  so this is a pure internal-rename refactor with no API changes.
- `v1.0.0` — gated on reproducing every main-paper figure from a fresh clone
  (Development Guide v2 §5.7).


## [0.6.0] — 2026-04-22

### Added — v2 pivot (Modules 1.1–3.3 + 4.1 + 5.2)

**Module 1.1 — CFD template and verification**
- `scalarTransportFoam` template (`0/T`, `DT` in `transportProperties`,
  `div(phi,T)` divScheme, `T` solver block).
- 1D advection-diffusion verification in `ooc_optimizer/cfd/scalar.py` with
  driver `scripts/run_scalar_verification.py` (analytic check at Pe ∈ {1,
  10, 100, 1000}; pass criterion L² < 2%).

**Module 1.2 — Parametric geometry**
- Two-inlet chamber generator across three topologies (`opposing`,
  `same_side_Y`, `asymmetric_lumen`).
- Topology-aware multi-block `blockMeshDict` generator
  (`ooc_optimizer/geometry/topology_blockmesh.py`) enforcing the v2 patch-
  naming contract (`inlet_drug`, `inlet_medium`, `outlet`, `walls`, `floor`,
  `frontAndBack`).
- 7D continuous parameter space: `W`, `d_p`, `s_p`, `theta`, `Q_total`,
  `r_flow`, `delta_W`.

**Modules 2.1/2.2 — Meshing + CFD orchestration**
- Two-step per-evaluation solve: `simpleFoam` → `scalarTransportFoam` on the
  frozen velocity.
- New metrics: `L2_to_target`, `grad_sharpness`, `monotonicity`.
- Target-profile callables (`linear_gradient`, `bimodal`, `step`, `custom`).

**Module 2.3 — Schema + logging**
- Pydantic v2 schema (`ooc_optimizer/config/schema.py`) with a v1-
  compatibility normaliser (`Q` → `Q_total`, defaults for `r_flow` /
  `delta_W` / `inlet_topology`).
- Extended JSONL evaluation logger carrying `inlet_topology`,
  `target_profile`, and both v2 and v1 objective keys.

**Module 3.1 — Bayesian optimisation**
- New `BORunner(config, pillar_config, H, topology, target_profile)`
  minimising `L2_to_target` subject to relaxed WSS (`τ ∈ [0.1, 2.0] Pa`) and
  no-dead-zone constraints.
- `run_multi_target_workflow()` — 24 configurations for the primary target
  then winning topology only for two secondary targets.
- Per-topology active-parameter masks so the GP never sees informationless
  inputs.

**Module 3.2 — Plotting**
- `ooc_optimizer/analysis/concentration_fields.py`: contour, residual,
  centreline, streamline overlay, convergence, winner grid.

**Module 3.3 (NEW) — Interpretability**
- `ooc_optimizer/interpretability/{sobol.py, gp_gradients.py, tolerance.py,
  pipeline.py}`.
- `scripts/run_interpretability.py` writes per-run `sobol.png`,
  `local_sensitivity.png`, `tolerance.png`, and `design_heuristics.md`.

**Module 4.1 — 3D CFD validation**
- `ooc_optimizer/geometry/topology_blockmesh.py::generate_blockmesh_dict_v2_3d`
  — z-extruded 3D meshes with `floor`/`ceiling` patches.
- `ooc_optimizer/validation/cfd_3d_v2.py::validate_winner_3d` runs
  simpleFoam + scalarTransportFoam in 3D for a BO winner and reports
  `L2_to_target_3d`, `L2_to_target_2d`, `L2_relative_delta`, and floor-WSS
  statistics.
- `ooc_optimizer/validation/compare_plots_v2.py` — three comparison figures
  (concentration residual, centreline overlay, WSS scatter + Bland-Altman).
- Driver: `scripts/run_3d_validation.py` (supports both a single BO state
  directory and a multi-target orchestrator summary).

**Module 5.2 — Repository + packaging**
- Hatchling-based `pyproject.toml` with pinned dependency ranges,
  dev/test/docs optional extras, ruff + mypy + pytest-cov config, and
  `tumor-chip` console script.
- `tumor_chip_design` compatibility namespace package re-exporting
  `ooc_optimizer` so `import tumor_chip_design` works after
  `pip install tumor-chip-design`.
- LICENSE (MIT), CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md,
  `.pre-commit-config.yaml`.
- GitHub workflows: `ci.yml` (ruff + mypy + pytest on Python 3.10 / 3.11 /
  3.12), `docs.yml` (mkdocs deploy), `release.yml` (PyPI + Zenodo).
- Issue / PR templates.
- `docker/Dockerfile` (OpenFOAM 2406 + Python 3.11 + package) and
  `docker/docker-compose.yml`.
- `mkdocs-material` docs skeleton (`docs/`) with index, installation,
  quickstart, concepts pages, and auto-generated API docs via
  `mkdocstrings`.
- JOSS companion paper skeleton in `paper/joss/`.
- Example configs + drivers under `examples/tumor_chip_linear_gradient/`,
  `examples/tumor_chip_bimodal/`, `examples/wss_uniformity/` (the latter
  retains the v1 WSS-uniformity worked example).

### Changed
- `ooc_optimizer/__version__` → `0.6.0`.
- Default `configs/default_config.yaml` now ships the v2 schema (7D
  parameter space, three topologies, three target profiles, passive-scalar
  diffusivity, interpretability block).


## [0.5.0] — 2026-04-09

- Hardened pillar meshing quality handling: removed degraded blockMesh
  fallback, added per-case meshing diagnostics/log files and checkMesh
  summaries, exposed snappy/mesh quality thresholds in config, enforced BO
  penalization + infeasible filtering for low-quality meshes
  (`mesh_ok=False`).


## [0.4.0] — 2026-04-02

- Implemented pillar-case geometry/meshing and analysis utilities: refactored
  CadQuery generator with validation and pillar-obstacle STL export,
  upgraded CFD setup for robust inlet/patch injection and pillar meshing
  workflow, and completed Module 3.2 plotting / report scripts
  (`convergence`, `comparison`, `wss_contours`, `run_analysis.py`) with
  smoke-tested figure generation.


## [0.3.0] — 2026-03-27

- Module 3.1 implemented: completed single-configuration BO runner and
  8-configuration orchestrator with constrained objective handling, Sobol
  initialisation, GP fitting, and run-state serialisation.


## [0.2.0] — 2026-03-20

- Module 1.1 complete: OpenFOAM solver verification.  Implemented Poiseuille
  analytical solution, blockMeshDict generator, OpenFOAM field parser,
  automated verification script, and mesh convergence study.


## [0.1.0] — 2026-03-13

- Scaffolded full project file structure reflecting all 5 pipeline phases
  and module dependencies.


[Unreleased]: https://github.com/ooc-loop/tumor-chip-design/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.6.0
[0.5.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.5.0
[0.4.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.4.0
[0.3.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.3.0
[0.2.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.2.0
[0.1.0]: https://github.com/ooc-loop/tumor-chip-design/releases/tag/v0.1.0
