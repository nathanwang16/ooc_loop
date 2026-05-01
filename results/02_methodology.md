# 2. Methodology

This document specifies the pipeline that produced every result in §4. The intent is operational fidelity: someone reading this document with access to the codebase should be able to reproduce any numerical result with no further reverse-engineering.

---

## 2.1 Pipeline overview

```
parametric design x  ──►  blockMesh (geometry)
                              │
                              ▼
                          simpleFoam (momentum, frozen flow)
                              │
                              ▼
                          scalarTransportFoam (passive tracer)
                              │
                              ▼
                          metric extraction → (L2, R²-to-lin, τ, f_dead, Re, Pe, AR)
                              │
                              ▼
                BoTorch BO ◄── ConstrainedEI over ModelListGP(obj, c1…c5)
                              │
                              ▼ (after convergence)
                  GP surrogate ──►  Sobol S_T  +  local sensitivity  +
                                    fabrication-tolerance intervals  +
                                    constraint-binding diagnostic
```

Each block is described below in the order of execution. Code paths cited are relative to the repository root.

## 2.2 Parametric geometry

The chamber is modelled as a rectangular conduit of fixed streamwise length `L = 10 mm`, chosen for compatibility with standard 3D cell-culture volumes and PDMS soft-lithographic fabrication. Seven continuous design parameters span the candidate device configurations:

| Parameter | Symbol | Unit | Range | Active in: |
|---|---|---|---|---|
| Chamber width | `W` | μm | [1500, 4500] | all |
| Total flow rate | `Q_total` | μL/min | [5, 200] | all |
| Inlet flow ratio | `r_flow` | — | [0.10, 0.97] | two-inlet trio |
| Inlet convergence half-angle | `θ` | deg | [15, 75] | two-inlet trio |
| Tongue offset | `δ_W` | — | [0.12, 0.48] | opposing only |
| Pillar diameter | `d_p` | μm | [100, 400] | pillar configs only |
| Pillar spacing | `s_p` | μm | [200, 1000] | pillar configs only |

The chamber height `H` is treated as a discrete parameter at `200 μm` or `300 μm`, fixed for a given optimization campaign. The aspect ratio `W/H` is constrained *a posteriori* to `≤ 15` (Section 2.5).

**Topology-dependent parameter masking.** Each topology activates a strict subset of the seven parameters; masked dimensions are pinned at the midpoint of their bounds and never moved by the BO acquisition. In code (`bo_loop.py:60-66`):

```python
if topology == "ladder":
    for name in ("theta", "r_flow", "delta_W", "d_p", "s_p"):
        active[PARAMETER_ORDER.index(name)] = False
```

So the bare-ladder (no pillars) optimisation is **2-D in `(W, Q_total)`**, not 7-D. Pillar configurations unmask `d_p` and `s_p`, making the pillar-ladder optimisation 4-D. The two-inlet topologies activate 4 (`asymmetric_lumen`, `same_side_Y`) or 5 (`opposing`, retains `delta_W`) dimensions.

The geometry generator (`ooc_optimizer/geometry/topology_blockmesh.py`) renders the parametric design as an OpenFOAM `blockMeshDict`. Each topology has its own block layout; the ladder, for example, partitions the `x = 0` face into N = 8 inlet patches each of height `W/N`, with prescribed concentration `C_k = (k+0.5)/N` (midpoint convention) for `k ∈ {0, ..., 7}` and uniform per-strip streamwise velocity `Q_total / (N · (W/N) · H) = Q_total / (W H)`.

The midpoint convention `C_k = (k+0.5)/N` instead of the more common endpoint convention `C_k = k/(N−1)` produces a uniform 38% L2 reduction at any fixed geometry by ensuring each strip's prescribed concentration equals the linear-target value at the strip's *centre*. The endpoint convention pins the concentration at the strip *edges*, leaving a `±1/(2N)` systematic offset at the edge strips that dominates L2 for any N. This is a free, generalisable improvement for any imposed-inlet gradient generator.

## 2.3 CFD pipeline

The CFD pipeline is two-stage and frozen-flow:

1. **Momentum (`simpleFoam`)** — steady incompressible Navier–Stokes. Default tolerances: residuals down to 1e-5 on `U`, 1e-6 on `p`. Up to 2000 iterations; convergence is checked by both residual drop and a separate divergence-of-`U` diagnostic. The case-dir naming includes PID + UUID6 fragment to prevent collisions when concurrent BO workers mint case directories from the same millisecond (`cfd/solver.py`; bug history in `tip.md`).
2. **Scalar transport (`scalarTransportFoam`)** — passive tracer on the frozen velocity field. The field is normalised to `C ∈ [0, 1]` with drug stream at `C = 1` and medium stream at `C = 0`. Diffusivity `D = 10⁻¹⁰ m²/s` (small-molecule drug surrogate). The discretisation scheme is `bounded Gauss upwind` for `div(phi, T)` (first-order upwind); a sharper `bounded Gauss limitedLinear 1` was tested but caused SIGFPE at high Pe in the 1D verification (recorded in `tip.md`). The upwind scheme contributes a small numerical-diffusion residual to L2; the residual budget is in `04_optimization_results.md` §2.4.

**Mesh density.** `ny_per_mm = 25` was specified in the diagnostic-findings configuration; the actual production mesh resolves at `ny_per_mm ≈ 19.6` (88 y-cells over W = 4495.6 μm at the H = 300 winner; verified from the case's `blockMeshDict`). Per-strip resolution at N = 8 is therefore ~11 cells per strip — defensible but tight.

**Verification.** Two independent verification studies live in the repo:

- **1-D advection–diffusion** (`scripts/run_scalar_verification.py`, `data/scalar_verification/`). The `scalarTransportFoam` solver on a 1-D channel with uniform velocity vs. the textbook analytic `C(ξ) = (1 − e^{−Pe(1−ξ)}) / (1 − e^{−Pe})`. L2 relative error: 0.064 % (Pe = 1, n = 100), 0.77 % (Pe = 10), 1.26 % (Pe = 100), 1.66 % (Pe = 1000, refined to n = 500 to suppress upwind diffusion to within tolerance). All four PASS at the 2 % bar.
- **Hagen–Poiseuille velocity / WSS** (`scripts/run_verification.py`, `data/verification/`). Three-level mesh refinement (nx = 100/200/400; ny = 10/20/40). Mean velocity error: 1.18 % → 0.39 % → 0.13 %. All levels PASS at 2 %.

Production meshes (`ny ≈ 25/mm`) sit on the converged plateau established by these studies.

## 2.4 Diagnostic-metric set

Every CFD evaluation produces a metric dictionary consumed by the BO and the comparison reports. The full set, with definitions:

| Metric | Definition | Used as |
|---|---|---|
| `L2` | $\|C - C_{\text{target}}\|_2 / \|C_{\text{target}}\|_2$ over the developed-flow region | objective |
| `R²-to-linear` | Least-squares regression `C(y) = a + b·(y/W)` over the depth-averaged profile; reports `R²`. | reported only for monotonic-linear targets; co-objective indicator |
| `tau_mean` | Mean wall shear stress on the chamber floor + ceiling, in Pa | constraint |
| `f_dead` | Fraction of the developed-flow region with `‖U‖ < 0.10 · U_mean_dev` (note: 10% of *region mean*, not 1% of `U_max`; cf. `metrics.py:237`) | constraint |
| `Re` | `ρ U_avg D_h / μ` with hydraulic diameter `D_h = 2WH/(W+H)` | constraint (laminar gate) |
| `Pe_streamwise`, `Pe_crossstream` | `U·ℓ/D` along and across flow | diagnostic |
| `aspect_ratio` | `W/H` | constraint |
| `monotonicity` | Fraction of consecutive bins with same-sign gradient along the target axis | diagnostic |
| `C_mean`, `C_std` | First and second moments of the C field over the developed region | diagnostic |
| `converged ∧ mesh_ok` | Booleans from solver residuals and mesh-quality checks | feasibility prerequisite |

Failure paths in `solver.py` and `metrics.py` map missing/NaN values to *finite* "deeply infeasible" sentinels (`Re = 1e6`, `aspect_ratio = 1e3`, `Pe_* = R² = 0.0`), not to NaN — early in the project a NaN-poisoned constraint GP crashed BoTorch's `ConstrainedExpectedImprovement` (root cause + fix in `tip.md`).

## 2.5 Constraint set

Five hard constraints define `X_feas`. Each is encoded as an independent GP constraint surrogate; the BoTorch acquisition is `ConstrainedExpectedImprovement` over a `ModelListGP(objective_GP, c_1, ..., c_5)`, which automatically marginalises constraint violation probability.

| # | Constraint | Threshold | Category |
|---|---|---|---|
| 1 | `tau_mean` ∈ [0.1, 2.0] Pa | both bounds | biology |
| 2 | `f_dead` ≤ 0.08 | upper | biology / flow uniformity |
| 3 | `Re` ≤ 100 | upper | safety / laminar gate |
| 4 | `aspect_ratio` ≤ 15 | upper | manufacturability (PDMS collapse) |
| 5 | converged ∧ mesh_ok | binary | numerical validity |

A point is feasible iff all five constraints hold; otherwise it is logged as infeasible and the GP for the violated constraint absorbs the boundary information. The acquisition function balances objective-EI against feasibility-probability automatically — the BO does not need to "know" which constraint was hit, only that the candidate produced a constraint vector outside the feasible polytope.

The constraint set was chosen by walking the categories: cell-viability shear window (1), no stagnant pockets (2), regime safety (3), fabrication (4), numerical validity (5). The justification for *each* is one of: a measurable lab quantity (1, 2, 4), a CFD-modelling assumption (3, 5). No optional constraints (e.g. pH, stagnation pressure, pumping power) are included — the rule was "every constraint must correspond to a documented failure mode in the literature or in the lab." Future projects with different cells or pumps may need different threshold values, but the *categories* are stable.

## 2.6 Bayesian optimization loop

**Topology as a categorical hyperparameter.** The pipeline runs *one independent BO loop per topology*, with a shared parameter space and constraint set. Cross-topology comparison is done by aggregating the per-topology bests and surrogates afterward. This pattern allows topology-specific masking (Section 2.2) without conditioning the GP kernel on a categorical input.

Per-topology BO settings:

- **Surrogate:** Matérn-5/2 GP on the input cube `[0, 1]^d_active`. Hyperparameters re-fitted by maximum marginal likelihood at every BO iteration. One GP per output (one objective + five constraints) under a `ModelListGP`.
- **Initialisation:** 24 Sobol-quasirandom points (`n_sobol_init = 24` in `examples/tumor_chip_linear_gradient/config.yaml`). Each point is evaluated with full CFD before the GP fit.
- **Acquisition:** `ConstrainedExpectedImprovement` (BoTorch). Optimised by Sobol-seeded L-BFGS multistart (16 starts, 256 raw samples).
- **Batch size:** 4 (parallel CFD workers). Total BO budget: 200 evaluations per topology = 24 init + 176 BO = 50 BO iterations × 4 batch.
- **Numerical detail:** the legacy `ConstrainedExpectedImprovement` (vs the newer `LogConstrainedExpectedImprovement`) emits a deprecation warning every iteration; both produce identical optima in our regime but the log form is recommended for future runs (see the paper revision plan, item F'.5).

The BO surrogate state is checkpointed every iteration to `bo_<topology>_<pillar>_H<H>/gp_model_state.pt`.

## 2.7 Interpretability triple — formulas

After each BO loop converges, four interpretability artefacts are computed from the trained surrogate. The first three use only the GP mean (no extra CFD calls); the fourth is purely descriptive.

### 2.7.1 Sobol total-effect indices

Let `μ(x)` be the trained GP-mean function over the input cube `[0, 1]^d_active`. The Sobol total-effect index for parameter `i` is

$$S_{T,i} = \frac{\mathbb{E}_{\mathbf{x}_{\sim i}}\bigl[\mathrm{Var}_{x_i}(\mu \mid \mathbf{x}_{\sim i})\bigr]}{\mathrm{Var}(\mu)}$$

estimated by Saltelli sampling at `n = 1024` (so `N = n(2d+2)` model evaluations on the GP mean). Implementation: SALib via `ooc_optimizer/analysis/sobol.py`. A faithful surrogate has `Σ S_T ≈ 1`; we use `Σ S_T > 1.5` as the trustworthiness threshold (anything above flags GP overfit). The audit table appears in `05_interpretability_findings.md` §3.

### 2.7.2 Local sensitivity at the optimum

At the BO optimum `x*` on the unit cube, the local sensitivity along axis `i` is

$$\bigl|\partial \mu / \partial x_{\text{norm}, i}\bigr|_{\mathbf{x} = \mathbf{x}^*}$$

computed by autograd through the GP mean. Reports the *first-order* effect at the optimum, complementing the Sobol indices' global picture. The two should largely agree; persistent disagreement flags either GP non-stationarity at the optimum or a discontinuous response surface (neither has been observed in our campaigns).

### 2.7.3 Fabrication-tolerance intervals

For each parameter `i`, find the largest `Δ⁻_i, Δ⁺_i ≥ 0` such that

$$\mu(\mathbf{x}^* - \Delta^-_i\,\mathbf{e}_i) \le 1.10\,\mu(\mathbf{x}^*) \quad \text{and} \quad \mu(\mathbf{x}^* + \Delta^+_i\,\mathbf{e}_i) \le 1.10\,\mu(\mathbf{x}^*)$$

with all other dimensions fixed. The bisection is performed on the GP mean (no extra CFD calls) and reports the result in physical units. Reading: "you can drift this parameter by Δ in either direction without losing more than 10% L2 quality." When the optimum sits at a bound, the corresponding side reports zero allowed drift — that asymmetry is itself a useful diagnostic (cf. `05_interpretability_findings.md` §4.2).

### 2.7.4 Constraint-binding diagnostic

For each of the five constraints, report at the BO optimum: the observed value, the threshold, and the slack (binding ↔ slack ≈ 0). No formula; the value is descriptive but is the most actionable single output of the entire pipeline (cf. `05_interpretability_findings.md` §4).

## 2.8 Reproducibility checklist

| Field | Value |
|---|---|
| OpenFOAM | v2406 |
| Python | 3.10 (conda env `ooc`) |
| BoTorch | current as of 2026-04-30 |
| GP kernel | Matérn-5/2 |
| GP-hyperparameter optimisation | maximum marginal likelihood, re-fit per BO iteration |
| Sobol initialisation seed | fixed per campaign (different seeds across the integration run vs. the H-sweep redo; see `04_optimization_results.md` §1.3) |
| Saltelli `n` for Sobol indices | 1024 |
| Hardware | Apple M4 (Mac mini, 2024), 32 GB unified memory |
| Total CPU-hours across reported campaigns | ≈ 6.5 CPU-h |
| OpenFOAM mesh density (production, ladder) | `ny_per_mm ≈ 20`, 88 y-cells at H = 300 winner |
| Per-eval CFD wall time | ≈ 12 s |

Wall-clock totals: 4-topology integration run = ~60 min; H-sweep (2 × 200 ladder evals) = 13 min 22 s; pillar-ablation BO = ~25 min. Verification studies = ~1 CPU-h. The pipeline is single-machine (no GPU; multiple CFD processes pinned to performance cores).

## 2.9 What the codebase does NOT do (yet)

Stated as a deliberate scoping decision, not as a limitation:

- **No 3-D CFD in the BO loop.** The 2-D approximation collapses the floor/ceiling boundary layers into the implicit `H` parameter; this is faithful for `W/H ≥ 5` (always satisfied at the optima reported here). A 3-D validation module (`ooc_optimizer/validation/cfd_3d_v2.py`) exists but was not run against the H = 300 winner in this cycle.
- **No transient CFD.** Every L2 is computed against the time-converged steady-state field; transients during chip startup or reagent switching are out of scope.
- **No multi-target BO.** Each campaign optimises a single L2 against a single target shape. Multi-target compromises (e.g. step + linear simultaneously) are not yet supported; they would require a Pareto-front formulation.
- **No experimental validation.** The pipeline has not been benchmarked against a fabricated chip. The R² = 0.990 is a CFD-vs-CFD-target metric; the chip-vs-CFD agreement is unknown until benchmark data exist.

These are documented in `06_translation_and_caveats.md` §3 alongside the publishable caveats.
