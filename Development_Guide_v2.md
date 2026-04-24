# tumor-chip-design: Development Guide (v2)

> **Status legend used throughout this document**
> `[UNCHANGED]` — identical to v1 (WSS-uniformity project)
> `[EXTENDED]` — same scaffold, adds new capability; v1 code is reused largely as-is
> `[REVISED]` — substantive rework; keeps the interface but replaces internals
> `[NEW]` — did not exist in v1
> `[REMOVED]` — present in v1, dropped in v2

---

## 0. Migration Summary

**Project pivot (v1 → v2).** From *"Bayesian optimization of chamber geometry for wall-shear-stress uniformity"* (WSS-uniformity) to *"Inverse design of tumor-on-chip chamber geometry for a target drug/tracer concentration profile, with interpretability output for fabrication tolerances"* (tumor-chip-design, a.k.a. A3′).

**Why pivot.** v1 has a known deflating critique (Di Carlo: a rectangular chamber with W ≫ H already gives near-uniform WSS, so the optimization is close to trivial). v2 reframes the pipeline around a harder forward problem (convection-diffusion on a prescribed velocity field, two inlets) and relocates the scientific contribution from "optimization finds a better chip" to "interpretability of a learned surrogate reveals which design parameters control the field and what tolerances labs need." This survives the Di Carlo critique because the deliverable is the *discovered design heuristic*, not the optimized geometry.

**What is preserved verbatim.**

| Layer | Component | Reuse |
|---|---|---|
| Team | Dev A / Dev B split and overall workload distribution | 100% |
| Pipeline shape | 5-phase linear chain (geometry → mesh → solve → BO → validation) | 100% |
| Geometry tooling | CadQuery parametric generator | ~70% of code reused |
| Meshing | `blockMesh` / `cfMesh` orchestration | ~95% |
| Momentum CFD | `simpleFoam` template case, boundary conditions, verification protocol | 100% |
| BO engine | BoTorch, Matérn 5/2 GP, Sobol init, `ConstrainedExpectedImprovement`, per-discrete-config sequential BO | 100% of mechanics; QoI and constraints change |
| Logging | YAML config + JSONL evaluation log | 100% (schema extended) |
| Fabrication stack | SLA mold → PDMS cast (Sylgard 184, 10:1) → corona bond → glass | 100% |
| Equipment | HeyGear Reflex (molds), Sovol SV08 (non-critical parts) | 100% |

**What changes at the module level.**

| Module | v1 | v2 status | Summary |
|---|---|---|---|
| 1.1 CFD template | simpleFoam, verify Poiseuille | `[EXTENDED]` | Add `scalarTransportFoam` verification against 1D advection-diffusion analytic solution |
| 1.2 Geometry generator | Single-inlet chamber, 5 cts + 2 disc params | `[REVISED]` | Two inlets (drug + medium), adds flow-ratio and inlet-topology discrete choice; 7 cts + 3 disc params |
| 2.1 Meshing | blockMesh / cfMesh on single-inlet STL | `[EXTENDED]` | Add second-inlet patch naming; otherwise unchanged |
| 2.2 CFD orchestrator | simpleFoam → extract WSS metrics | `[REVISED]` | Two-step solve (velocity, then passive scalar); new metrics (L² to target, gradient sharpness, monotonicity) |
| 2.3 Config + logger | YAML schema, JSONL log | `[EXTENDED]` | Add scalar BCs, target profile spec, new QoI fields |
| 3.1 BO loop | Minimize CV(τ) s.t. WSS constraints | `[EXTENDED]` | Same mechanics; minimize L²(C, C_target) s.t. flow & no-dead-zone constraints |
| 3.2 Analysis + plots | WSS contours, BO convergence | `[EXTENDED]` | Adds concentration-field contours alongside WSS |
| **3.3 Interpretability** | — | `[NEW]` | **Sobol indices, GP gradient sensitivities, per-parameter tolerance intervals — this is the novel scientific output** |
| 4.1 3D CFD validation | simpleFoam 3D | `[EXTENDED]` | Adds 3D scalar transport; reports L²(C,C_target) in 3D vs. 2D |
| 4.2 Fabrication | 1 inlet + 1 outlet, 5 chips | `[EXTENDED]` | 2 inlets + 1 outlet, chip count revised below |
| 4.3 Flow experiments | Food dye + fluorescein washout RTD | `[REVISED]` | **Primary**: steady-state fluorescein concentration-field imaging. **Secondary**: washout RTD retained as independent velocity-field check |
| 5.1 Figures | WSS-centric figure list | `[REVISED]` | Concentration-field-centric; adds interpretability figure |
| **5.2 Code packaging** | Basic README + deps list | `[REVISED]` | **Full open-source compliance (pyproject, src-layout, CI, docs, DOI, CITATION.cff) — see §6** |
| 5.3 Manuscript | Biomicrofluidics / Lab on a Chip, WSS title | `[REVISED]` | Retargeted; emphasis on interpretability contribution |

**What is removed.** The WSS-uniformity pipeline is *not deleted*. It is retained as `examples/wss_uniformity/` in the repo — a second worked example that demonstrates the framework's generality (same engine, different scalar: wall shear instead of concentration). This also preserves all v1 work as a publishable auxiliary contribution (software-methods paper).

---

## 1. Project Overview (Non-Technical) `[REVISED]`

This project builds an open-source, reproducible pipeline that designs, simulates, fabricates, and validates microfluidic tumor-on-chip devices. The goal is not to ship a single "best" chip, but to answer a methodology question: *given a target concentration profile (for example, a drug or nutrient gradient across a tumor culture chamber), which chamber geometry produces it, and — more importantly — which geometric features actually matter?*

The pipeline: a parametric CAD generator produces candidate chamber geometries, a CFD solver computes the resulting velocity field and the concentration field of a passive tracer, a Bayesian optimizer searches for the geometry that best matches a target profile, and a post-hoc interpretability layer analyzes the learned surrogate to report (a) which parameters dominate the field and (b) how tightly each parameter needs to be held during fabrication. The optimized geometry is then 3D-printed as a mold, cast in PDMS, and imaged with a fluorescent tracer to confirm the predicted concentration field.

The primary scientific deliverables are (1) a set of design heuristics and fabrication-tolerance intervals for tumor-on-chip chambers with spatial concentration control, (2) an open-source software toolkit that other labs can reuse for related inverse-design problems, and (3) a peer-reviewed manuscript.

---

## 2. Target Scope `[REVISED]`

**Single biological framing:** tumor-on-chip, 3D culture in hydrogel in a central chamber, drug (or nutrient analog) delivered via one inlet and buffer via the other, with spatially-shaped exposure across the chamber. Organoid itself is *not modeled* — cell consumption is treated as a sensitivity analysis variable, not a simulation input. The "tracer" in simulation is a passive scalar; in experiments it is sodium fluorescein.

**Target profile class for v2 optimization runs:**

1. Linear gradient along the chamber long axis.
2. Bimodal profile (high at two regions, low in the middle) — mimics "rim + rim" exposure patterns.
3. Step profile (sharp transition at x = L/2).

Run the full pipeline on each target class. Showing that the same pipeline finds plausibly different optimal geometries for different targets is the core methodological claim.

**Optimization fidelity:** 2D CFD during BO; 3D CFD + physical imaging for post-hoc validation of the winner for each target class.

**Validation experiments:** steady-state fluorescein concentration-field imaging (primary), plus fluorescein washout RTD retained from v1 as an independent velocity-field check.

**Discrete variable handling:** enumerate pillar configurations × chamber heights × inlet topologies; run independent BO per combination. No mixed-integer optimization.

**Total computational budget (revised upward):** ~480 CFD runs across 24 discrete configurations (4 pillar layouts × 2 heights × 3 inlet topologies), at 60 evaluations each (20 Sobol + 40 BO iters), times 3 target profiles run back-to-back on the single best-performing topology for each target → effective budget ≈ 1,500 forward solves. Scalar transport is ~1.5× the cost of `simpleFoam` alone. Estimated total: ~12 h on 8 cores. Fits in a weekend.

---

## 3. Team Structure and Ownership `[UNCHANGED]`

Two developers: Dev A (expert programmer + 3D printing access) and Dev B (intermediate programmer). Dev A owns modules requiring deep OpenFOAM/BoTorch expertise or physical equipment. Dev B owns geometry generation, configuration, results analysis, interpretability, and code packaging. The ownership assignments per module are listed in each module spec below — most are identical to v1, with Dev B picking up the new interpretability module (3.3) and the expanded code-packaging module (5.2).

---

## 4. Architecture Overview `[REVISED]`

The pipeline is still a linear chain, now six stages instead of five (scalar transport added):

1. **Parametric Geometry Generator** → fluid-domain STL + mold STL
2. **Automated Meshing**
3. **Momentum CFD** (`simpleFoam`) → velocity field
4. **Passive Scalar Transport** (`scalarTransportFoam` on the frozen velocity) → concentration field `C(x,y)`
5. **Bayesian Optimization Loop** wrapping stages 1–4 as a black box
6. **Interpretability + Validation + Experiment**

Stages 3 and 4 are decoupled deliberately: the scalar field doesn't feed back into momentum (passive tracer, no reaction, no buoyancy), so solving them sequentially is correct and cheaper. Running them in a single coupled solver would waste compute.

A shared YAML configuration passes parameters between stages. Every stage is fully automated — no human intervention per evaluation.

---

## 5. Open-Source Standards and Repository Structure `[NEW]`

This section is the concrete answer to "make it usable by other people." The repository targets compliance with JOSS review criteria ([https://joss.readthedocs.io/en/latest/review_criteria.html](https://joss.readthedocs.io/en/latest/review_criteria.html)) and FAIR4RS principles.

### 5.1 Repository layout

```
tumor-chip-design/
├── pyproject.toml              # PEP 517/518 packaging, single source of truth
├── README.md                   # Quickstart, install, citation, badges
├── LICENSE                     # MIT (permissive, standard for scientific Python)
├── CITATION.cff                # Machine-readable citation metadata
├── CHANGELOG.md                # Keep a Changelog format, SemVer
├── CONTRIBUTING.md             # How to report issues, submit PRs, run tests
├── CODE_OF_CONDUCT.md          # Contributor Covenant v2.1
├── .pre-commit-config.yaml     # ruff, black, mypy, end-of-file-fixer
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # pytest + ruff + mypy on PRs; matrix over Python 3.10/3.11/3.12
│   │   ├── docs.yml            # build + deploy docs to gh-pages
│   │   └── release.yml         # tag → PyPI + Zenodo DOI
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   └── tumor_chip_design/
│       ├── __init__.py
│       ├── geometry/           # Module 1.2
│       │   ├── __init__.py
│       │   ├── chamber.py      # parametric geometry generator
│       │   ├── pillars.py      # pillar placement logic
│       │   ├── validation.py   # isValid, feature-size checks
│       │   └── export.py       # STL export with unit conversion
│       ├── cfd/                # Modules 1.1, 2.1, 2.2
│       │   ├── __init__.py
│       │   ├── templates/      # OpenFOAM case templates (data, installed via package)
│       │   ├── meshing.py      # blockMesh / cfMesh orchestration
│       │   ├── momentum.py     # simpleFoam wrapper
│       │   ├── scalar.py       # scalarTransportFoam wrapper   [NEW]
│       │   └── metrics.py      # QoI extraction (WSS and concentration)
│       ├── optimization/       # Module 3.1
│       │   ├── __init__.py
│       │   ├── objectives.py   # target profiles, L² loss   [NEW]
│       │   ├── constraints.py  # flow, dead-zone, monotonicity
│       │   └── bo.py           # BoTorch loop
│       ├── interpretability/   # Module 3.3   [NEW entire subpackage]
│       │   ├── __init__.py
│       │   ├── sobol.py        # global sensitivity
│       │   ├── gp_gradients.py # local sensitivity via GP derivatives
│       │   └── tolerance.py    # tolerance-interval estimation
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py       # pydantic models for YAML validation
│       │   └── logger.py       # JSONL evaluation logger
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── plotting.py     # publication-quality figure utilities
│       │   └── experimental.py # image analysis for validation chips
│       └── cli.py              # `tumor-chip` entry point (Typer)
├── tests/
│   ├── conftest.py
│   ├── test_geometry.py        # unit tests on constraints, STL validity
│   ├── test_cfd_verification.py
│   ├── test_metrics.py
│   ├── test_objectives.py
│   ├── test_bo.py              # uses a synthetic test function, not real CFD
│   ├── test_interpretability.py
│   ├── integration/
│   │   └── test_end_to_end.py  # tiny BO run on no-pillar config (~2 min)
│   └── fixtures/               # reference STLs, reference CFD results
├── docs/
│   ├── mkdocs.yml              # mkdocs-material
│   ├── index.md
│   ├── installation.md
│   ├── quickstart.md           # 15-min end-to-end walkthrough
│   ├── concepts/
│   │   ├── parametric-geometry.md
│   │   ├── scalar-transport.md
│   │   ├── inverse-design.md
│   │   └── interpretability.md
│   ├── tutorials/
│   │   ├── 01-geometry-sweep.ipynb
│   │   ├── 02-single-bo-run.ipynb
│   │   ├── 03-interpretability.ipynb
│   │   └── 04-fabrication-walkthrough.md
│   ├── api/                    # auto-generated via mkdocstrings
│   └── publications.md         # list citing the software
├── examples/
│   ├── tumor_chip_linear_gradient/
│   │   ├── config.yaml
│   │   ├── run.py
│   │   └── expected_output/
│   ├── tumor_chip_bimodal/
│   └── wss_uniformity/         # v1 project retained as an exemplar
│       ├── config.yaml
│       ├── run.py
│       └── README.md
├── docker/
│   ├── Dockerfile              # OpenFOAM 11 + Python 3.11 + package
│   └── docker-compose.yml
├── scripts/
│   ├── run_all_configs.sh      # orchestrates 24 discrete configs
│   └── rebuild_figures.py      # regenerates all paper figures from logs
└── paper/
    ├── manuscript.tex
    ├── figures/                # generated, not committed
    └── references.bib
```

### 5.2 Packaging and dependency management

- **`pyproject.toml`** as the single source of truth. No `setup.py`, no `setup.cfg`. Build backend: `hatchling`.
- **`src/` layout** (avoids the common pitfall of accidentally importing from the repo root).
- **Pinned ranges, not exact versions**, in `pyproject.toml` — e.g. `botorch >= 0.11, < 0.13`. Exact versions live in `requirements-lock.txt` generated by `pip-compile`.
- **Docker image** for reproducibility of the OpenFOAM side. `openfoam/openfoam11-paraview510` as base, package layered on top. `docker-compose` exposes a ready-to-use shell.
- **Python versions supported**: 3.10, 3.11, 3.12. CI matrix tests all three.

### 5.3 Code quality

- **Formatter + linter**: `ruff` (configured in `pyproject.toml`, both `ruff format` and `ruff check`). One tool, fast, covers what `black` + `isort` + `flake8` used to.
- **Type checker**: `mypy --strict` on `src/`. Tests are allowed to be loose.
- **Docstrings**: NumPy style, enforced via `ruff` (pydocstyle rules).
- **Pre-commit**: `pre-commit install` after clone runs ruff + mypy + end-of-file-fixer on staged files.

### 5.4 Testing

- **Framework**: `pytest` with `pytest-cov`.
- **Target coverage**: ≥ 80% on `src/`, excluding `cfd/templates/` (data files) and `cli.py` (thin wrapper).
- **Test tiers**:
  - Unit tests (< 1 s each): pure Python logic — geometry validation, metric math, constraint encoding.
  - Integration tests (< 30 s each): exercise meshing and a tiny `simpleFoam` + `scalarTransportFoam` case on a ~1k-cell mesh.
  - End-to-end smoke test (< 5 min): a two-iteration BO run on the simplest no-pillar configuration. Skipped by default, runs nightly in CI.
- **No live CFD in unit tests.** Mock the solver subprocess in unit tests; use the integration tier for the real thing.

### 5.5 Documentation

- **`mkdocs-material`** for a static site. Deployed to GitHub Pages via `docs.yml`.
- **API docs auto-generated** via `mkdocstrings` from NumPy-style docstrings — no duplication.
- **Tutorials as executed notebooks**: CI re-runs them on every push (fails if they break). Uses `jupyter nbconvert --execute`.
- **A 15-minute quickstart** in `quickstart.md` that a new user can follow end-to-end using the Docker image, producing a real CFD result. This is the single most important onboarding page.

### 5.6 Citability

- **`CITATION.cff`** at repo root (GitHub renders a "Cite this repository" button automatically).
- **Zenodo integration**: tagged releases trigger a Zenodo archive with a DOI. README badge displays the DOI.
- **JOSS submission in parallel with the main manuscript** — the two target different audiences (JOSS: software; main manuscript: scientific contribution).

### 5.7 Versioning and release

- **Semantic Versioning 2.0.0**. v0.x during development; v1.0.0 gated on the end-to-end pipeline reproducing all paper figures from a fresh clone.
- **`CHANGELOG.md`** in Keep a Changelog format, updated on every merge to main.
- **Releases**: tag on main → GitHub Actions builds wheels, uploads to PyPI, archives to Zenodo, builds Docker image.

### 5.8 What community standards we're deliberately *not* adopting (honest list)

- **No conda-forge recipe** initially. PyPI + Docker covers the realistic user base.
- **No benchmarking suite** (e.g., `asv`). Not useful for a CFD pipeline where wall-clock is dominated by the external solver.
- **No type stubs for OpenFOAM calls**. They're subprocess invocations; typing them is noise.
- **No i18n**. English only.
- **No formal governance model** until there are external contributors. A `MAINTAINERS.md` listing Dev A and Dev B suffices.

---

## 6. Module Specifications

Modules in chronological development order. Modules within the same phase can be developed in parallel.

---

### Phase 1: Foundation (Weeks 1–2)

---

#### Module 1.1 — CFD Template and Solver Verification `[EXTENDED]`

**Owner:** Dev A

**What stays from v1.** All of the momentum verification: `simpleFoam` template, 2D single-cell-thick slab with `empty` front/back, Poiseuille verification in a straight rectangular channel, mesh convergence study at 1×/2×/4× refinement, convergence criteria (residuals < 1e-6 for U and p). The existing `τ_floor = 6 μ U_avg / H` approximation is retained as a **diagnostic** for the retained WSS constraint, not as the primary QoI.

**What is added.** A second verification problem for `scalarTransportFoam`:

- Solve 1D steady advection-diffusion on the Poiseuille velocity field with `C = 1` at inlet, `C = 0` initial, Peclet number Pe = U·L/D varied across {1, 10, 100, 1000} spanning the regime expected for small-molecule drugs in our chamber.
- Compare against the analytic solution `C(x) = (1 - exp(Pe·x/L)) / (1 - exp(Pe))`.
- Pass criterion: L² error < 2% on a 100-cell mesh.

This establishes that the scalar solver is trustworthy *before* connecting it to the BO loop. Without this step, any anomaly in BO output has two possible causes (optimizer bug vs. solver bug); doing the verification up front eliminates one of them.

**Boundary conditions for scalar transport (new, added to the v1 BC table):**

| Patch | Concentration (T or scalar field) |
|---|---|
| inlet_drug | `fixedValue` (C = 1) |
| inlet_medium | `fixedValue` (C = 0) |
| outlet | `zeroGradient` |
| walls, floor | `zeroGradient` (no flux; cell consumption modeled via sensitivity, not in forward solve) |
| frontAndBack | `empty` |

**Deliverables (extends v1 deliverables):**
- Verified `scalarTransportFoam` template case.
- Documented verification plot: analytic vs. simulated `C(x)` at four Pe.
- `diffusivity` input parameter in config: 1×10⁻¹⁰ m²/s default (small-molecule drug surrogate); range 1×10⁻¹¹ to 1×10⁻⁹ m²/s documented for sensitivity runs.

---

#### Module 1.2 — Parametric Geometry Generator `[REVISED]`

**Owner:** Dev B

**What stays from v1.** CadQuery-based approach, the `generate_chip(params) → (fluid_stl, mold_stl)` interface, internal units (μm, converted to mm on export), mold construction (bounding box minus fluid domain, 2 mm wall thickness, glass-slide alignment pins), geometry validation checks (`isValid()`, feature size ≥ 100 μm), unit-test structure. The pillar placement logic (even grid, center positions from R×C formula) is reused verbatim.

**What changes.** The chip now has **two inlets and one outlet**. The chamber is still rectangular (L = 10 mm fixed, W continuous), but the inlet topology is a new discrete axis. Three inlet topologies:

- `opposing`: drug inlet at (0, W/2 − δ), medium inlet at (0, W/2 + δ), both on the x = 0 face, separated by a tongue of PDMS of width 2δ.
- `same_side_Y`: Y-junction on the x = 0 face merging drug and medium into a single inlet; the BO then controls the mixing ratio via flow rates.
- `asymmetric_lumen`: Ayuso-2020-style — drug delivered via a lumen running along one long side of the chamber, medium via the opposite short edge. This is the most biologically faithful option.

**Continuous parameters (v2):**

| Variable | Symbol | Range | Units | Change from v1 |
|---|---|---|---|---|
| Chamber width | W | 500–3000 | μm | unchanged |
| Pillar diameter | d_p | 100–400 | μm | unchanged |
| Pillar gap | s_p | 200–1000 | μm | unchanged |
| Inlet taper angle | θ | 15–75 | ° | unchanged |
| Total flow rate | Q_total | 5–200 | μL/min | unchanged (renamed from Q) |
| **Flow ratio** | r_flow = Q_drug / Q_total | 0.1–0.9 | — | **new** |
| **Inlet separation** | δ / W (only for `opposing`) | 0.1–0.45 | — | **new** |

Dimensionality: 7 continuous. For `same_side_Y` and `asymmetric_lumen` topologies δ is not applicable — the BO runs in 6D for those, 7D for `opposing`. This is handled by per-topology parameter masks in the config (BoTorch can handle this via separate BO runs per topology, which is already how we handle discrete configs).

**Discrete parameters (v2):**

| Variable | Levels | Change |
|---|---|---|
| Pillar configuration | {none, 1×4, 2×4, 3×6} | unchanged |
| Chamber height | {200, 300} μm | unchanged |
| **Inlet topology** | {opposing, same_side_Y, asymmetric_lumen} | **new** |

Total discrete combinations: 4 × 2 × 3 = 24. Up from 8 in v1.

**Fixed parameters:** L = 10 mm, inlet channel widths = 500 μm, μ = 0.001 Pa·s, ρ = 1000 kg/m³. Plus new: `diffusivity D` (see Module 1.1). All unchanged from v1 in structure.

**Geometry validation checks (extends v1):**
- All v1 checks retained.
- New: **patch-naming contract** — the STL must export patches named exactly `inlet_drug`, `inlet_medium`, `outlet`, `walls`, `floor`, `frontAndBack`. This is what Module 2.1 expects.
- New: **reachability check** — both inlets must connect to the chamber via the fluid domain (cheap: check that the fluid volume is a single connected component).

**Deliverables:**
- `src/tumor_chip_design/geometry/chamber.py` exposing `generate_chip(params, topology) → (fluid_stl, mold_stl)`.
- Verified STLs for all 24 discrete configurations at representative continuous values (committed to `tests/fixtures/`).
- Unit tests for each topology, each pillar config, edge cases at parameter bounds.
- Migration note in the repo: v1's single-inlet generator is retained as `examples/wss_uniformity/geometry.py`, pointing at a frozen older revision of `chamber.py`.

---

### Phase 2: Pipeline Integration (Weeks 3–4)

---

#### Module 2.1 — Automated Meshing Pipeline `[EXTENDED]`

**Owner:** Dev A (with Dev B providing STL inputs)

**What stays from v1.** Two-path strategy (blockMesh for no-pillar, cfMesh preferred for with-pillar, snappyHexMesh fallback), one-cell-thick z for 2D, `empty` front/back, `checkMesh` gate, penalty-return fallback for mesh failures, cell-count targets (~12k no-pillar, ~30k with-pillar).

**What changes.** The patch-naming logic is extended to recognize `inlet_drug` and `inlet_medium` as separate patches (v1 had one `inlet`). The scalar solver reads these patch names from the BC dict to apply different `C` values.

No other changes. Meshing is unaware of whether the solver will use one scalar or two momentum components — the mesh is identical.

**Deliverables:** mostly unchanged from v1. Add a regression test: meshing a two-inlet STL produces two distinct inlet patches in the resulting `polyMesh/boundary` file.

---

#### Module 2.2 — CFD Run Automation and Metric Extraction `[REVISED]`

**Owner:** Dev A

**What stays from v1.** The overall shape of `evaluate(params) → metrics_dict`. Error handling: penalty value on solver divergence or mesher failure. 5-minute per-evaluation timeout. JSON-serializable output. Subprocess-based OpenFOAM invocation.

**What changes.** The orchestrator now runs **two** solves in sequence per evaluation:

1. `simpleFoam` on the meshed geometry with momentum BCs (unchanged from v1). Emits `U` and `p` fields.
2. `scalarTransportFoam` reading the converged `U` field (via `changeDictionary` or the standard OpenFOAM frozen-flow pattern), with scalar BCs (C = 1 at `inlet_drug`, C = 0 at `inlet_medium`, zeroGradient elsewhere). Emits `C` field.

Wall-clock: step 1 is ~20–60 s, step 2 is ~15–40 s. Together ~35–100 s per evaluation. Budget estimate (§2) accounts for this.

**New metrics extracted from each evaluation:**

| Metric | Definition | Role |
|---|---|---|
| `L2_to_target` | `‖C - C_target‖₂ / ‖C_target‖₂` on the chamber floor | **Primary objective (minimize)** |
| `grad_sharpness` | Mean of `‖∇C‖` over chamber floor, normalized to max possible gradient | Diagnostic |
| `monotonicity` | Fraction of chamber length over which `∂C/∂x` has consistent sign (when a monotonic target is set) | Diagnostic |
| `f_dead` | Fraction of floor where `‖U‖ < 0.1 × mean ‖U‖` | **Constraint (must be < 0.05)** — unchanged from v1 |
| `tau_mean` | Area-weighted mean floor WSS, from `τ = 6μU/H` | **Constraint (must be in [0.1, 2.0] Pa)** — v1 bounds relaxed |
| `converged_U`, `converged_C` | Booleans | Gate: either false → penalty |

The WSS bounds relaxed relative to v1 because drug transport is far less WSS-sensitive than endothelial biology; the constraint is now "flow is reasonable and not pathological," not "endothelial-cell-physiological."

**Target profile definition (new).** `C_target(x, y)` is a callable defined in the config. Provided callables:

- `linear_gradient(axis='x', c_high=1.0, c_low=0.0)`: `C_target = c_low + (c_high - c_low) × (x/L)` along `axis`.
- `bimodal(peaks=(L/4, 3L/4), width=L/10)`: sum of two Gaussian bumps.
- `step(x0=L/2, sharpness=0.01·L)`: sigmoid transition at `x0`.
- `custom(fn)`: user-supplied Python callable.

The target is evaluated on the CFD mesh cell centers at post-processing time, so L² is computed on matched sampling.

**Deliverables:**
- `evaluate(params, topology, H, target_profile_name) → metrics_dict`, fully automated.
- JSONL log per evaluation including input parameters, solver outputs, achieved `C(x,y)` summary statistics, convergence flags, and wall-clock timings.
- Robust error handling tested on known-bad geometries (zero volume, non-watertight STL, disconnected fluid domain).

---

#### Module 2.3 — Configuration Schema and Logging `[EXTENDED]`

**Owner:** Dev B

**What stays from v1.** YAML config file, central parameter bounds, solver settings, paths, JSONL evaluation logger, replay/inspect utility.

**What is added.** Pydantic models (`src/tumor_chip_design/config/schema.py`) replace raw YAML dict access. This gives:

- Schema validation on load (caught-at-startup rather than mid-BO-run errors).
- IDE autocomplete on config access.
- Automatic docstring → docs propagation for the config reference page.

New schema fields:

- `diffusivity: float` (with pinned default 1e-10 m²/s)
- `target_profile: TargetProfileConfig` (with subtypes `LinearGradient | Bimodal | Step | Custom`)
- `inlet_topology: Literal["opposing", "same_side_Y", "asymmetric_lumen"]`
- Extended `bounds` block including `r_flow` and `delta_W`
- `interpretability: InterpretabilityConfig` with `sobol_n_samples: int = 1024`, `tolerance_loss_tolerance: float = 0.1`

**Deliverables:** pydantic schema, sample config `examples/tumor_chip_linear_gradient/config.yaml`, migration script `scripts/migrate_v1_config.py` that reads a v1 YAML and emits a v2 YAML defaulting to `linear_gradient` target and `opposing` topology.

---

### Phase 3: Optimization (Weeks 4–5)

---

#### Module 3.1 — Bayesian Optimization Loop `[EXTENDED]`

**Owner:** Dev A

**What stays from v1 (verbatim).** BoTorch as framework. Matérn 5/2 kernel with learned length scale and output scale. Sobol initialization. `ConstrainedExpectedImprovement` acquisition. One GP per constraint. Per-discrete-config independent BO runs (embarrassingly parallel). Selection of overall winner by best feasible objective across all runs. Model + data serialization after every run.

**What changes.**

- **Objective:** `L2_to_target` (instead of `CV(τ)`).
- **Constraints:** (a) `0.1 - tau_mean ≥ 0`? No — constraint is `tau_mean ≥ 0.1` and `tau_mean ≤ 2.0`. (b) `0.05 - f_dead ≥ 0`. Formally three constraint GPs, same as v1.
- **Budget per configuration:** 60 evaluations (20 Sobol + 40 BO), up from 50. More dimensions justify more samples.
- **Total configurations:** 24 (was 8). But not all 24 are run for every target profile; the overall workflow runs all 24 for *one* target profile (linear gradient) to establish the topology ranking, then runs only the best-performing topology for the other target profiles. This is the cost-control choice that keeps total budget tractable.
- **Design rationale carried forward:** single-objective with hard constraints is retained. The v1 rationale against weighted-sum multi-objective still applies.

**Deliverables:** largely unchanged from v1; add convergence logging for scalar residuals alongside momentum residuals, and add serialization of the target profile alongside the BO run for later reproducibility.

---

#### Module 3.2 — Results Analysis and Visualization `[EXTENDED]`

**Owner:** Dev B

**What stays from v1.** BO convergence curves per configuration. Parameter heatmaps across configurations. Constraint-satisfaction scatter plots. Baseline comparison (for v2, the baseline is "no pillars, 90° taper, symmetric opposing inlets, r_flow = 0.5"). Matplotlib style, publication dpi, vector formats.

**What is added.** Concentration-field plots replace WSS contour plots as the key figure:

- `C(x, y)` contour map for the BO winner of each target profile.
- `C_achieved − C_target` residual map (shows where the optimizer did well vs. badly).
- 1D line plots: `C(x, W/2)` vs. x — achieved vs. target — for each winner.
- Streamline overlays where relevant (shows why a given geometry produces a given field).

**Deliverables:** `src/tumor_chip_design/analysis/plotting.py` with a function per figure, all consuming the JSONL log. `rebuild_figures.py` script at repo root regenerates all paper figures in < 5 minutes from the log files. No figure is generated manually in Illustrator or Inkscape — this is a hard rule for reproducibility.

---

#### Module 3.3 — Interpretability Analysis `[NEW]`

**Owner:** Dev B

**Purpose.** This module is the primary scientific contribution of v2. The BO gives you *a* geometry that matches a target field. That's not very useful to other labs by itself. What's useful is the answer to: *which geometric parameters actually control the field, and how much can each be perturbed before the match degrades?* That's what this module computes.

**Three analyses per BO run:**

1. **Global sensitivity via Sobol indices.**
   - Tool: `SALib` library (established, maintained, plays well with BoTorch).
   - Sample the trained GP surrogate (not the CFD directly — orders of magnitude cheaper) at ~1024 Sobol points.
   - Compute first-order indices `Sᵢ` and total-order indices `S_Tᵢ` for each continuous parameter.
   - Output: bar chart of `Sᵢ` and `S_Tᵢ` per parameter, with bootstrap confidence intervals.
   - Interpretation: parameters with low `S_Tᵢ` are the ones labs can relax.

2. **Local sensitivity at the optimum via GP gradients.**
   - At the BO winner `x*`, compute `∇μ(x*)` analytically from the GP (Matérn 5/2 has a closed-form gradient; BoTorch exposes this).
   - Normalize per-parameter: `∂μ/∂xᵢ × (range(xᵢ))` gives a comparable sensitivity across parameters with different units.
   - Output: ordered list of parameters by local sensitivity at the optimum.
   - Interpretation: complements Sobol — Sobol is global (averaged over the design space), this is local (at the specific optimum).

3. **Tolerance intervals.**
   - For each parameter `xᵢ`, starting at the optimum `x*`, compute the perturbation `Δxᵢ` that causes `L² loss` to degrade by 10% (configurable).
   - Method: bisection on the trained GP mean along each axis; validate the top-3 most tolerant and top-3 least tolerant by actually re-running CFD at `x* ± Δxᵢ`.
   - Output: table of tolerance intervals per parameter, with validation-CFD error bars on the top-3 from each end.
   - Interpretation: this is the direct fabrication-tolerance spec — for each parameter, "here's how precisely you need to hold it."

**Design decision — why GP-based not CFD-based.** Running 1024 extra CFD evaluations per BO winner is infeasible (order ~15 hours). Sampling the trained GP is ~seconds. The GP is already a validated surrogate over the sampled region (we measured its accuracy during BO), so using it here is principled. The validation-CFD check on the top/bottom-3 tolerances is the safety net: if GP-predicted sensitivity disagrees with real CFD by more than the GP error bar, we know to retrain or expand sampling.

**Deliverables:**
- `src/tumor_chip_design/interpretability/{sobol.py, gp_gradients.py, tolerance.py}`.
- Three figures: Sobol bar chart, local-sensitivity ranking, tolerance interval table.
- A short "design heuristics" text output per target profile, auto-written as markdown: "For target T, the dominant parameters are X and Y; Z and W can be held loosely; fabrication tolerance on X must be tighter than ±N%."

---

### Phase 4: Validation (Weeks 6–7)

---

#### Module 4.1 — 3D CFD Validation `[EXTENDED]`

**Owner:** Dev B

**What stays from v1.** `snappyHexMesh` on the winner's STL, 500k–1M cells, 5-layer boundary layer, `simpleFoam` 3D, 2D vs. 3D comparison (scatter + Bland-Altman) on WSS as a sanity check that the 2D approximation didn't mislead the optimizer.

**What is added.** 3D scalar transport on the 3D winner. Report `L2_to_target` in 3D vs. 2D — this is the key validation: if the 2D-optimized geometry also matches the target in 3D, the 2D-during-BO choice was correct. Run for one representative winner per target profile (not all winners — budget).

**Deliverables:** 3D `simpleFoam` + `scalarTransportFoam` results for BO winners of each target profile; 2D-vs-3D comparison figures for both WSS and concentration.

---

#### Module 4.2 — Fabrication Protocol `[EXTENDED]`

**Owner:** Dev A

**What stays from v1 (verbatim).** SLA mold printing (HeyGear Reflex, 50 μm layer, face-down orientation, IPA wash + UV post-cure). PDMS casting (Sylgard 184, 10:1, degas 30 min, cure 65 °C × 2 h). Corona bonding to glass. Biopsy-punching fluidic ports. Tygon tubing. Luer-lock syringe-pump connection. Priming with DI water. Optional silane release and thin epoxy mold coating.

**What changes — minor.** The fabrication protocol handles two-inlet chips:
- 3 punch locations per chip (was 2): drug inlet, medium inlet, outlet.
- Two tubing lines to the syringe pump (two channels or two separate pumps).
- Separate priming of each inlet before connecting to common chamber (prevents air entrapment in the PDMS tongue between opposing inlets).

**Chip lineup for v2 (revised from v1's five-chip lineup):**

| Chip | Geometry | Replicates | Purpose |
|---|---|---|---|
| A | BO winner for linear-gradient target | 2 | Demonstrate pipeline works, target 1 |
| B | BO winner for bimodal target | 2 | Demonstrate generality, target 2 |
| C | BO winner for step target | 2 | Demonstrate generality, target 3 |
| D | Baseline (no pillars, 90° taper, symmetric opposing inlets, r_flow=0.5) | 2 | Comparator |
| E | Deliberate off-tolerance chip (most-sensitive parameter perturbed by the predicted tolerance amount) | 1 | **Validates the interpretability prediction** |

Total: 9 chips, ~3 days of fabrication. The extra budget is necessary because chip E is what validates the interpretability output — without it, Module 3.3's tolerance claims are unverified.

**Deliverables:** 9 assembled, leak-tested chips; fabrication log; dimensional check of printed molds against CAD (optical microscope, measure 3 critical dimensions per chip).

---

#### Module 4.3 — Flow Experiments and Data Analysis `[REVISED]`

**Owner:** Dev A

**What stays from v1.** Syringe-pump flow control, UV lamp + blue-filter camera imaging setup, OpenCV-based image analysis, ROI-based intensity extraction, normalization against background and saturated reference.

**What changes — primary experiment is different.**

**Experiment 1 (PRIMARY, new framing) — Steady-state fluorescein concentration-field imaging.**

Protocol: Prime the chip with DI water. Set total flow rate to `Q_total*` (from the BO-winner config). Switch drug inlet to 0.1 mg/mL fluorescein in PBS, medium inlet to plain PBS. Wait 5× the theoretical mean residence time for steady state. Capture a long-exposure, well-lit, top-down image of the chamber under UV illumination. Normalize against (a) a dark frame and (b) a saturated-chamber reference (both inlets fluorescein).

Analysis: normalize pixel intensity `I(x,y) → C_norm(x,y) ∈ [0, 1]`, assuming fluorescein fluorescence is linear in concentration at the chosen dye concentration (verified by a calibration curve at the start of the session, standard protocol). Compare `C_norm(x,y)` to the CFD-predicted `C(x,y)` for the same chip. Primary quantitative metric: L² distance between measured and predicted fields. Secondary: line profiles along the chamber centerline.

**Experiment 2 (SECONDARY, retained from v1) — Fluorescein washout RTD.**

Identical protocol to v1. This is an *independent* check on the velocity field: if the measured RTD matches the CFD-predicted RTD, the underlying momentum solve was correct, which in turn anchors the concentration-field agreement. Without this, a poor concentration-field match is ambiguous (was the flow wrong, or the diffusion wrong?). With this, we can attribute.

**Quantitative comparison metrics:**

| Metric | Method | Expected |
|---|---|---|
| Field L² (measured vs. CFD) | Pixel-wise on normalized images | < 0.15 for chips A/B/C; > 0.25 for baseline |
| Centerline profile match | `C(x, W/2)` line plots overlaid | Visual + L² |
| Interpretability prediction for chip E | Predicted field degradation within ±20% of measured | Validates Module 3.3 |
| RTD half-life CV across ROIs | from Experiment 2 | Consistent with 2D CFD (< 15% error) |

**Deliverables:** raw images + extracted concentration fields for all 9 chips; field-vs-CFD comparison figures (grid of 9 subplots: measured / predicted / residual per chip); the interpretability validation figure for chip E.

---

### Phase 5: Documentation and Publication (Weeks 8–10)

---

#### Module 5.1 — Paper Figures `[REVISED]`

**Owner:** Both (Dev A: CFD and experimental figures; Dev B: BO and interpretability figures)

Figure list:

1. **Pipeline schematic** — block diagram: spec → CadQuery → 2 OpenFOAM solves → BoTorch → interpretability → SLA → PDMS → fluorescence imaging.
2. **Solver verification** — Poiseuille WSS and 1D advection-diffusion analytic vs. simulated.
3. **Geometry / parameter-space overview** — the three inlet topologies, annotated.
4. **BO convergence** — objective vs. iteration, per topology, per target profile.
5. **Concentration-field winners** — 3-column grid: target, achieved 2D, achieved 3D. One row per target profile.
6. **Residual fields** — `C_achieved − C_target` maps. Highlights where each design struggles.
7. **Interpretability — Sobol indices** — bar chart per target profile.
8. **Interpretability — tolerance intervals** — table + figure showing predicted tolerance vs. measured degradation.
9. **Experimental validation** — grid of 9 chips: measured field / predicted field / residual.
10. **Chip E — interpretability validation** — predicted vs. measured degradation from the designed off-tolerance perturbation.

The WSS figures from v1 do not appear in the main paper; they go to the supplementary material alongside the retained WSS-uniformity example.

---

#### Module 5.2 — Code Packaging and Repository `[REVISED]`

**Owner:** Dev B

This module is the implementation of §5. See §5 for the detailed specification. The deliverables for this module are:

- Fully populated repository matching the layout in §5.1.
- CI green on Python 3.10/3.11/3.12 for lint, type-check, tests, and docs build.
- Docker image published to `ghcr.io/<org>/tumor-chip-design:<version>`.
- Tagged `v1.0.0` release on GitHub, archived to Zenodo, DOI displayed in README.
- JOSS submission draft in `paper/joss/` alongside the main manuscript.
- The 15-minute quickstart tutorial executes end-to-end on the Docker image without manual intervention.

---

#### Module 5.3 — Manuscript `[REVISED]`

**Owner:** Both

**Target venue.** *Lab on a Chip* (strongest fit — applied methodology + experimental validation). Fallback: *Biomicrofluidics* or *Microfluidics and Nanofluidics*.

**Working title.** *"Interpretable inverse design of tumor-on-chip chambers: learned surrogates reveal which geometric features control drug gradients and by how much."*

**Structural emphasis.** Lead with the interpretability output, not the BO pipeline. A reviewer's first question will be "why geometry and not valves" — the manuscript needs to answer this in the introduction, briefly and honestly: geometry-based inverse design is not claimed to replace valves for high-throughput screening; the contribution is in understanding *which geometric features matter* for the passive, single-chip class of devices (Ayuso-2020-style) that are already in active use in many labs.

**Companion paper.** JOSS submission on the software, targeted for co-publication with the main paper. Short (~4 pages), focuses on software architecture, reproducibility, and extensibility to other inverse-design problems (the retained WSS example is the extensibility demonstration).

---

## 7. Development Schedule `[REVISED]`

| Week | Dev A | Dev B | Milestone |
|---|---|---|---|
| 1 | 1.1: OpenFOAM env, Poiseuille verify, **add scalarTransportFoam verify (1D analytic)** | 1.2: CadQuery two-inlet chamber, all 3 topologies, STL export | Foundation verified on both momentum and scalar |
| 2 | 1.1: finalize templates; begin 2.1 patch-naming update | 1.2: mold geom, constraint validation, unit tests | Both pillars complete |
| 3 | 2.1: meshing with two-inlet patches; 2.2 momentum side | 2.3: pydantic schema, extended JSONL logger | Meshing + config ready |
| 4 | 2.2: add scalarTransportFoam orchestration + frozen-flow pattern + new metrics | 2.3: finish logger; begin 3.2 plotting framework | End-to-end `evaluate()` works |
| 5 | 3.1: BO over 24 configs × 1 target (linear gradient); then winner-topology × 2 more targets | 3.2: BO convergence + concentration-field plots | Optimization complete |
| 6 | 4.2: print molds, cast PDMS, assemble 9 chips | **3.3: Sobol + GP gradients + tolerance intervals (NEW)** | Interpretability + fabrication in parallel |
| 7 | 4.3: steady-state fluorescein imaging + washout RTD on all 9 chips | 4.1: 3D CFD validation (winner of each target profile) | Physical data collected |
| 8 | 4.3: analyze exp data; 5.1 CFD + exp figures | 5.1: interpretability figures; 3.3 finalize | All figures drafted |
| 9 | 5.3: manuscript draft | **5.2: full community-standards packaging (pyproject, CI, docs, Docker, Zenodo)** | Repo publishable + draft ready |
| 10 | 5.3: revise + submit | 5.3: JOSS companion; 5.2: tag v1.0.0 | Submission-ready |

**Buffer.** If fabrication or experiments slip in weeks 6–7, week 8 absorbs the delay and writing shifts by 1 week. The new interpretability module (3.3) is scheduled for Dev B in week 6 in parallel with Dev A's fabrication — this is deliberate: it decouples the two developers' critical paths.

---

## 8. Honest risks, carried from the initial critique

Three risks to hold in mind throughout. Not hidden here; if reviewers ask, we have an answer.

- **"Why geometry, not valves?"** Answer: single-use, passive-integrated chips are already in use (Ayuso 2020, many others); valves and geometry are not competing, they target different use cases. Our contribution is understanding geometry-based chambers, which had no formal inverse-design treatment.
- **"Is the 2D approximation valid for concentration fields?"** Validated empirically in Module 4.1 (2D vs. 3D CFD). If 2D L² error vs. 3D exceeds 10%, we report this honestly and move BO to 3D for the affected topology. Budget absorbs this; it pushes a weekend into a week.
- **"Target profile is arbitrary."** Partially true. We run three distinct target classes (linear, bimodal, step) to show pipeline generality. If a reviewer asks why those three, the answer is: they span monotonic / multimodal / discontinuous shapes, covering the qualitative space. Physiological justification for specific targets (e.g., match an in-vivo tumor profile) is in the discussion as future work.

---

*End of Development Guide v2. For v1 (WSS-uniformity) see the v1 guide, now retained in `examples/wss_uniformity/v1_development_guide.md`.*
