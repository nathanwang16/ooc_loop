# Tumor-on-Chip Inverse Design — Project Report

**Repository:** `ooc_loop` (Bayesian-optimization pipeline for tumor-on-chip chamber geometry).
**Reporting date:** 2026-04-26.
**Scope of this report:** the diagnostic + topology-pivot work completed in this work cycle, with a forward-projected results section for the integration run currently executing under a delegated background agent.

> Sections marked **[PROVISIONAL]** describe a run that is still in progress at time of writing; the structure and expected findings are recorded so the writeup skeleton is ready, but numbers should be replaced from the actual artifacts when the agent completes.

---

## 1. Introduction

The goal of this project is to design a microfluidic tumor-on-chip chamber that exposes 3D-cultured cells to a prescribed concentration profile of a soluble drug (or surrogate tracer). The flagship target is a **linear concentration gradient across the cell chamber**, motivated by its use as a single-chip dose-response test: each cell sees a distinct concentration depending on its position, replacing a multi-chip dose curve with a single device.

The pipeline supports the design as an inverse problem. A parametric chamber geometry is meshed with OpenFOAM `blockMesh`, momentum is solved with `simpleFoam`, scalar transport with `scalarTransportFoam`, and the resulting concentration field is scored against the linear-gradient target by a normalised-RMS L2 metric. A Bayesian-optimization loop (BoTorch, Matérn 5/2 GP, ConstrainedExpectedImprovement) searches the geometry/flow design space, with hard biological/manufacturability constraints encoded as separate constraint GPs.

Three two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`) were the original design space. Initial campaigns suggested the L2 surface was unusually flat and that the optimum was sitting close to a geometric floor that no parameter setting could break. **The work in this cycle confirms mechanistically that the original three topologies are physically incapable of producing a linear x-gradient at any setting (mass conservation), screens five new topology candidates, prototypes the most promising one (a Whitesides-style stacked ladder), and integrates it into the production stack alongside two new dimensionless-physics constraints (laminar Re, PDMS-collapse aspect ratio).**

---

## 2. Methodology

### 2.1 Diagnostic phase — testing the three existing topologies

The pipeline was already operational with `opposing` (two short-side inlets at x=0 with a tongue), `same_side_Y` (Y-junction split into two half-height inlets), and `asymmetric_lumen` (Ayuso-2020-style lumen on a side wall). A 600-evaluation BO campaign (200 evals × 3 topologies, pillar=`none`, H=200 µm) was run against `linear_gradient axis=x` with `c_high=1.0, c_low=0.0`. Each evaluation runs full CFD (~12 s wall) and reports `L2_to_target`, `monotonicity`, `f_dead`, `tau_mean`.

A **case-dir collision bug** was discovered mid-run (148 of the first 387 evals raised `FileExistsError` when concurrent BO workers minted case directories from the same millisecond). The fix added `os.getpid()` and a `uuid4().hex[:6]` fragment to the case-dir name (`ooc_optimizer/cfd/solver.py`); the patched run completed with zero collisions over 39 minutes.

Following the patched campaign, three first-principles diagnostics ran (scripts in `scripts/`):

- `diagnostic_baseline.py` — single CFD evaluation at the default-config baseline parameters against several target shapes (linear axis=x, linear axis=y, step axis=y at multiple sharpness levels) to localise the failure mode.
- Closed-form computation of the uniform-field L2 floor against the linear target: `L2_uniform ≈ 0.585`.
- Comparison against the BO winner from the patched campaign (L2 = 0.6343, only 8% above the uniform-field floor).

### 2.2 Mass-conservation argument

For steady incompressible flow with two coflow inlets and no through-wall scalar source, advective-flux conservation across any vertical slice forces `<C>_y(x) ≈ r_flow` along the entire chamber. **Any linear x-gradient requires `<C>_y(x) = x/L`, which varies with x — mathematically incompatible with this constraint.** The L2 floor at 0.585 is the geometric lower bound for a uniform field at C=r_flow against the linear target; no design can break it. The chamber's *natural* concentration structure is transverse (y-direction stratification + diffusive smoothing), not streamwise.

### 2.3 Topology screening — five novel candidates

Five replacement topologies were proposed and rendered as 3-D schematic figures (`scripts/visualize_topology_candidates.py`):

- **A `ladder`** — N stacked y-strips at x=0, each with prescribed C_k. Whitesides 2000 / Dertinger 2001 ancestry. Target axis=y.
- **B `christmas_tree`** — binary mixer tree feeding 8 stacked chamber inlets. Target axis=y.
- **C `side_injection`** — main x=0 medium inlet plus K drug ports along y=0 with x-varying flow rates. **Only candidate that produces a real axis=x gradient** (the original target axis).
- **D `permeable_wall`** — chamber floor replaced by a permeable membrane with graded permeability against a drug reservoir. Target axis=x.
- **E `counter_flow`** — drug at x=0, medium at x=L, side-wall outlets. Steady-state existence is conditional on Re; high vortex-shedding risk.

Each was reasoned about from first principles (Pe, Re, mass conservation, manufacturability) and ranked. The literature was surveyed in parallel (Whitesides 2000, Dertinger 2001, Ayuso 2020, Yang 2020, Hashemi-Tilehnoee 2025, Borrvall–Petersson 2003) to confirm: (i) the y-axis convention is canonical, (ii) Christmas-tree + Kriging surrogate is published prior art (Yang 2020 RSC Adv 10:13799) but not in tumor-on-chip context, (iii) Sobol sensitivity on a CFD-trained surrogate of a tumor-chip gradient chamber has no published precedent — that aspect of this project is genuinely novel.

### 2.4 Phase 1 — single-shot CFD on candidate A (ladder)

A standalone diagnostic (`scripts/diagnostic_ladder_baseline.py`) was written that bypasses the production CFD path: it generates a custom `blockMeshDict` with N inlet patches at x=0, programmatically writes `0/U`, `0/p`, `0/T` BCs for all N inlets (uniform per-inlet U_x, prescribed C_k per inlet), runs `blockMesh + simpleFoam + scalarTransportFoam`, reads the C field, and computes L2 against `linear_gradient axis=y`.

Three conventions for assigning C_k to the N strips were tested:

- **Endpoint**, `C_k = k/(N−1)` — strip 0 at C=0, strip N−1 at C=1; centres mismatch the linear target by ±1/(2N).
- **Midpoint**, `C_k = (k+0.5)/N` — strip *centres* land exactly on the linear target.
- **N sweep** at fixed convention to test mesh-resolution scaling.

### 2.5 Phase 2 — Sobol-quasirandom scan over (W, Q_total)

Used as a stand-in for full BO (the production-pipeline integration is the work item below). 32 Sobol-quasirandom evaluations over `W ∈ [1500, 4500] µm` and `Q_total ∈ [5, 200] µL/min` at fixed N=8 endpoint convention. Implemented in `scripts/diagnostic_ladder_scan.py`; reuses the Phase-1 mesh + BC writers.

### 2.6 Production integration (in progress at time of writing)

A delegated Opus 4.7 background agent is currently executing the full integration plan at `/Users/lemon/.claude/plans/proceed-with-all-the-tingly-map.md`:

- **New diagnostic metrics** (`ooc_optimizer/cfd/metrics.py:extract_v2_metrics`): Reynolds (`Re = ρ·U_avg·D_h/µ`), Péclet streamwise/crossstream (`Pe = U·ℓ/D`), `aspect_ratio = W/H`, and `R²-to-linear` (least-squares fit of binned C against `a + b·ξ`, reported only for monotonic linear targets).
- **New hard BO constraints** (`ooc_optimizer/optimization/bo_loop.py`): `Re_max ≤ 100` (laminar gate), `aspect_ratio_max ≤ 15` (PDMS-collapse gate, relaxed from a literal 10 per W/H ≤ 15 user choice). The existing `tau_mean ∈ [0.1, 2.0]` and `f_dead ≤ 0.08` constraints are retained. The BO acquisition (ConstrainedExpectedImprovement over ModelListGP) auto-adapts to 5 constraint GPs.
- **Ladder topology integrated into production stack** (`ooc_optimizer/geometry/topology_blockmesh.py:_bm_ladder` lifted from the Phase-1 prototype; `ooc_optimizer/cfd/solver.py:_setup_case` dispatches to multi-inlet BC writers when `BlockMeshResult.inlet_names` is non-empty).
- **Target axis flipped to y** (`examples/tumor_chip_linear_gradient/config.yaml`): the original `axis: x` was identified as physically infeasible for two-inlet topologies; changed to `axis: y` matching the Whitesides/Dertinger canonical convention.
- **Cross-topology comparison report extended** (`examples/tumor_chip_linear_gradient/run.py:_write_comparison_report`): per-topology S1/ST Sobol cross-tabulation, constraint-feasibility table, and per-topology median diagnostic metrics.

The integration run is a 4-topology BO campaign (`opposing`, `same_side_Y`, `asymmetric_lumen`, `ladder`) at 200 evals/topology, axis=y target, under the new 5-constraint feasibility set. Wall time estimate: 60–90 minutes.

Topologies B–E remain prototyped only as schematic figures; each requires 1–1.5 days of additional mesh/BC engineering and is on the roadmap, not in this cycle.

---

## 3. Results

### 3.1 Original-three campaign (axis=x linear gradient)

| Topology | Best feasible L2 | Notes |
|---|---|---|
| `opposing` | **0.6343** (winner) | wedged near uniform-field floor 0.585; `f_dead = 0.0796 / 0.08` constraint-active |
| `asymmetric_lumen` | 0.7351 | all 200 evals feasible |
| `same_side_Y` | 0.8296 | `r_flow` wedged at upper bound (0.97) |

**Closed-form uniform-field L2 floor:** ≈ 0.585. **The BO winner sits 8% above this floor.** Combined with `monotonicity = 0.500` (chance level along x) and `C_mean = 0.672`, the result is unambiguously consistent with "BO is optimising toward the *most uniform field possible*", which is the analytical limit of what two-inlet coflow can produce against an x-gradient target.

### 3.2 Phase-1 single-shot CFD on ladder (axis=y target)

| Configuration | L2_to_target_axis_y | Improvement vs axis=x BO winner |
|---|---|---|
| Old `opposing` BO winner (axis=x reference) | 0.6343 | — |
| Ladder N=4, midpoint | 0.1337 | 4.7× |
| Ladder N=8, **endpoint** | 0.1756 | 3.6× |
| Ladder N=8, **midpoint** | **0.1097** | 5.8× |
| Ladder N=16, endpoint | 0.1423 | 4.5× |
| Ladder N=16, midpoint | 0.1091 | 5.8× (saturates here) |

Two key findings: (i) **changing the strip-concentration convention from endpoint to midpoint drops L2 from 0.176 → 0.110**, a 38% improvement from BC choice alone, no optimization needed; (ii) **L2 saturates around 0.109 with midpoint convention even at N=16**. The analytical floor for an N=8 step against a linear ramp (within-strip variation only) is `1/(2N√3) / (1/√3) = 1/(2N) ≈ 0.063`. The observed 0.110 is 1.76× this floor; the gap is dominated by numerical mesh diffusion (upwind scheme + ny_per_mm=25).

### 3.3 Phase-2 Sobol scan over (W, Q_total)

32 evaluations, all feasible, L2 range [0.1720, 0.1759] — **a 0.004 spread**, essentially flat. Best at `W = 1605 µm, Q_total = 10.17 µL/min`, only 2.1% better than the 0.1756 baseline at default (W=3000, Q=50). The scan demonstrates that **(W, Q_total) tuning is inert** for the ladder topology in the current high-Pe regime — the streams flow stratified end-to-end and chamber-flow parameters neither help nor hurt the imposed inlet ladder. The L2 floor is set entirely by the discrete-ladder approximation error and numerical diffusion.

### 3.4 [PROVISIONAL] 4-topology integration BO campaign

Expected results once the background agent completes:

- **Best feasible L2 by topology** (axis=y target, 5-constraint feasibility):

| Topology | Expected best L2 | Active dimensions | Notes |
|---|---|---|---|
| `ladder` (new) | **~0.10–0.11** | W, Q_total (2D) | should win clearly |
| `opposing` | ~0.6–0.8 (worse than axis=x) | W, theta, Q_total, r_flow, delta_W (5D) | mass-conservation forbids axis=y from two-inlet coflow giving a clean ramp; ~equal failure mode |
| `same_side_Y` | ~0.7–0.9 | W, theta, Q_total, r_flow (4D) | similar |
| `asymmetric_lumen` | ~0.7–0.9 | W, d_p, s_p, theta, Q_total, r_flow (6D, depending on pillar) | the side-lumen helps marginally, still bounded |

- **Constraint-binding analysis (expected):**
  - `aspect_ratio_max ≤ 15` will bind at H=200 µm: caps W at 3000 µm; the existing topologies' W will redistribute below this cap.
  - `Re_max ≤ 100` will not bind at our flow rates (`Re ≈ 1–10` even at upper Q_total bounds); functions as a safety rail.
  - `f_dead ≤ 0.08` retains its existing role; ladder is expected to score `f_dead ≈ 0` (no dead zones with N parallel streams).
  - `tau_mean ∈ [0.1, 2.0]` retains its role.

- **Cross-topology Sobol indices (expected):**
  - `ladder`: nearly all sensitivity in W and Q_total, with both showing low S_T (flat surface, consistent with the Phase-2 scan).
  - `opposing`: `r_flow` and `theta` should dominate S_T as before.
  - `same_side_Y`: similar to opposing but without `delta_W`.
  - `asymmetric_lumen`: `theta` and `Q_total` dominant.

- **Headline**: ladder wins by ~5–6× on L2 against axis=y. The other three "lose" on the axis=y target the same way they lost on axis=x — they are physically the wrong topology for any imposed-direction linear gradient with two coflow inlets.

These projected numbers should be replaced from the actual integration-run artifacts when available. The agent will write a separate `integration_run_findings.md` next to the campaign results.

---

## 4. Interpretation and Discussion

The diagnostic phase produced a clean, mechanistic answer to a question the BO results alone could not have settled: **the L2 ≈ 0.585 floor on the original three topologies is geometric, not algorithmic.** No length of BO campaign, no widening of bounds, no re-tuning of acquisition function, and no mesh refinement could have broken it. The two-inlet coflow geometry, in the laminar regime mandated by the cell-biology constraints, can only produce a transverse step (axis=y stratification with diffusive smoothing) — and the original target was specified along x. Mass conservation is the invariant that makes this conclusion robust.

The pivot to a y-axis linear-gradient target on a Whitesides-style ladder topology resolves the structural mismatch and produces an L2 of 0.110 — a 5.8× improvement and below the uniform-field floor that the original three topologies could not break. The Phase-2 Sobol scan is, in itself, an interpretability result: it shows that within the achievable regime, chamber-flow parameters are inert and the residual L2 floor is set by step-quantization plus numerical diffusion. **The natural follow-up is not "more BO over (W, Q_total)" but BO over the per-inlet `C_k` vector** (8 monotonic continuous parameters that can absorb the residual nonlinearity from cross-stream diffusion), or, alternatively, mesh refinement and a sharper advection scheme to drop the numerical-diffusion contribution.

Topologies B–E (christmas_tree, side_injection, permeable_wall, counter_flow) were screened with first-principles physical analysis but not implemented. Each has a "hidden parameter" set whose value determines whether the C-field is actually linear (vs. plateau, vs. saturating, vs. step-with-impingement-noise). Each requires non-trivial mesh and BC engineering; they are recorded as a roadmap, not abandoned.

The integration work currently in flight makes the ladder a first-class citizen of the BO pipeline, adds the two missing dimensionless-physics constraints (`Re_max`, `aspect_ratio_max`), and adds five new diagnostic metrics (`Re`, `Pe_streamwise`, `Pe_crossstream`, `aspect_ratio`, `R²_to_linear`) so the dimensional analysis of the design surface is surfaced in every evaluation log. The cross-topology comparison report — extended with per-topology Sobol S1/ST tables and constraint-slack diagnostics — is intended to be the publication-figure root.

**Honest caveats.** The Phase-1 ladder result depends on assuming a stable laminar Pe (≈ 4·10⁷ at default conditions); at very low Q the streams begin to merge by transverse diffusion and the ladder degrades. The midpoint convention is a free 38% improvement that any future user of this topology should not have to rediscover; it is now the production default. The 0.110 L2 is *not* the achievable minimum — mesh-resolution refinement, scheme upgrade, and per-inlet C_k optimization can reasonably target ~0.04, but each is an engineering investment.

The literature survey confirmed that the ladder + Kriging-surrogate design is published prior art (Yang 2020 RSC Adv 10:13799; Hashemi-Tilehnoee 2025 Lab Chip), but **the combination of (i) tumor-on-chip 3D-culture chamber, (ii) per-topology Sobol sensitivity comparison on the CFD-trained GP, and (iii) explicit fabrication-tolerance reporting from the surrogate is novel.** That combination is the project's defensible contribution; the BO and ladder are well-engineered scaffolding around it.

---

## 5. Asset directories — for write-up and poster

All paths are relative to the repository root `/Users/lemon/Desktop/ooc_loop/`.

### 5.1 Schematic figures (5 candidate topologies — ready)
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/A_ladder_N8.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/B_christmas_tree.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/C_side_injection_K8.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/D_permeable_wall.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/E_counter_flow.png`

### 5.2 Phase-1 diagnostic artifacts (ladder single-shot — ready)
- `examples/tumor_chip_linear_gradient/data/diagnostic/baseline_metrics_axis_x.json` — opposing baseline against axis=x linear (L2 ≈ 1.0)
- `examples/tumor_chip_linear_gradient/data/diagnostic/baseline_metrics_axis_y.json` — opposing baseline against axis=y linear (L2 ≈ 1.32)
- `examples/tumor_chip_linear_gradient/data/diagnostic/baseline_metrics_step_y_sharp0p05.json` and `..._sharp0p20.json`
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder/baseline_metrics.json` — N=8 endpoint (L2 = 0.1756)
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder/case/` — full converged OpenFOAM case (open `case.foam` in ParaView)

### 5.3 Phase-2 scan artifacts (Sobol over W, Q_total — ready)
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/results.jsonl` — per-eval rows
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/summary.json` — best, bounds, summary stats
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/heatmap.png` — (W, Q_total) → L2 colour map with best-point star

### 5.4 Original-three campaign artifacts (axis=x, archived)
- `examples/tumor_chip_linear_gradient/data/results/_aborted_run_20260426_000743/` — pre-patch run with FileExistsError noise
- `examples/tumor_chip_linear_gradient/data/results/evaluations_<topology>_none_H200.jsonl` — patched run, full per-eval logs
- `examples/tumor_chip_linear_gradient/data/results/optimization_summary.json` — winner + per-topology best-feasible
- `examples/tumor_chip_linear_gradient/data/results/bo_<topology>_none_H200/` — per-topology BO state and GP checkpoints
- `examples/tumor_chip_linear_gradient/data/results/diagnostic_findings.md` — the in-depth analysis document (authoritative source for §1–§3.3 of this report)

### 5.5 Integration-run artifacts (axis=y, in progress — paths are predicted)
- `examples/tumor_chip_linear_gradient/data/results/_aborted_run_axisX/` — old axis=x artifacts archived here by the agent
- `examples/tumor_chip_linear_gradient/data/results/evaluations_<topology>_none_H200.jsonl` — new 4-topology axis=y eval logs
- `examples/tumor_chip_linear_gradient/data/results/optimization_summary.json` — new winner identification
- `examples/tumor_chip_linear_gradient/data/results/bo_<topology>_none_H200/` — new per-topology BO state (4 directories, including `bo_ladder_none_H200/`)
- `examples/tumor_chip_linear_gradient/data/results/comparison_report.md` — extended cross-topology report (S1/ST table, constraint slack, diagnostic metrics)
- `examples/tumor_chip_linear_gradient/data/results/integration_run_findings.md` — the agent's own honest summary
- (interpretability figures from `scripts/run_interpretability.py`) — typically heatmaps, sensitivity bars, tolerance intervals next to the BO state directories

### 5.6 Source / scripts (for the methodology section of a writeup)
- `scripts/visualize_topology_candidates.py` — generates §5.1
- `scripts/diagnostic_baseline.py` — closed-form uniform-field check
- `scripts/diagnostic_ladder_baseline.py` — Phase-1 single-shot
- `scripts/diagnostic_ladder_scan.py` — Phase-2 Sobol scan
- `scripts/run_optimization.py` — production BO entry
- `scripts/run_interpretability.py` — Sobol on GP surrogate
- `examples/tumor_chip_linear_gradient/run.py` — comparison-report writer

### 5.7 Project documentation
- `tip.md` — bug history (case-dir collision fix; numerical diffusion notes)
- `README.md` — revision history (will receive a one-line entry on integration completion)
- `Development_Guide_v2.md` — architectural reference; the BO + interpretability framing in §1 of this report follows it

---

## 6. What's next

Once the integration agent completes:

1. Confirm the four-topology BO results match the [PROVISIONAL] expectations in §3.4. Replace the provisional table with actual numbers.
2. Inspect the new `comparison_report.md` for the cross-topology Sobol cross-tab — this is the headline figure for the design-heuristics paper.
3. Consider the **per-inlet `C_k` 8-D BO campaign** as the next active campaign (lets the optimizer absorb numerical-diffusion bias by tuning inlet concentrations slightly off the linear ladder).
4. Topologies B–E remain in the roadmap (1–1.5 days each); `side_injection` is the only path to a real axis=x gradient if that target is ever required, and may be worth an investment for a follow-up paper.
5. Density-based topology optimization (Borrvall–Petersson 2003 in Brinkman-penalised Navier–Stokes) is a publishable separate-project follow-up — explicit roadmap in `diagnostic_findings.md`.
