# Tumor-on-Chip Inverse Design — Project Report

**Repository:** `ooc_loop` (Bayesian-optimization pipeline for tumor-on-chip chamber geometry).
**Reporting date:** 2026-04-26 (post H-sweep extended test; final).
**Scope of this report:** diagnostic phase + topology pivot + production integration + extended H-sweep, all completed. **Headline finding: ladder topology at H=300 μm achieves L2 = 0.0671 (R² = 0.990) — a 9.45× improvement over the original axis=x BO winner.**

---

## 1. Introduction

The goal of this project is to design a microfluidic tumor-on-chip chamber that exposes 3D-cultured cells to a prescribed concentration profile of a soluble drug (or surrogate tracer). The flagship target is a **linear concentration gradient across the cell chamber**, motivated by its use as a single-chip dose-response test: each cell sees a distinct concentration depending on its position, replacing a multi-chip dose curve with a single device.

The pipeline supports the design as an inverse problem. A parametric chamber geometry is meshed with OpenFOAM `blockMesh`, momentum is solved with `simpleFoam`, scalar transport with `scalarTransportFoam`, and the resulting concentration field is scored against the linear-gradient target by a normalised-RMS L2 metric. A Bayesian-optimization loop (BoTorch, Matérn 5/2 GP, ConstrainedExpectedImprovement) searches the geometry/flow design space, with hard biological/manufacturability constraints encoded as separate constraint GPs.

Three two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`) were the original design space. Initial campaigns suggested the L2 surface was unusually flat and that the optimum was sitting close to a geometric floor that no parameter setting could break. **The work in this cycle (i) confirms mechanistically that the original three topologies are physically incapable of producing a linear x-gradient at any setting (mass conservation), (ii) screens five new topology candidates, (iii) prototypes the most promising one (a Whitesides-style stacked ladder) in a stand-alone diagnostic, (iv) integrates it into the production stack alongside two new dimensionless-physics constraints (laminar Re, PDMS-collapse aspect ratio) and five new diagnostic metrics (Re, Pe streamwise/crossstream, aspect ratio, R²-to-linear), (v) runs a 4-topology BO comparison against an `axis=y` linear-gradient target where the ladder wins decisively (L2 = 0.082 at H=200 μm), and (vi) confirms via an extended H-sweep that opening the chamber height to 300 μm relaxes the binding `aspect_ratio_max=15` cap and drops the achievable L2 a further 18% to 0.0671.**

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

### 2.6 Production integration (completed)

The integration plan at `/Users/lemon/.claude/plans/proceed-with-all-the-tingly-map.md` was executed in two passes:

- **New diagnostic metrics** (`ooc_optimizer/cfd/metrics.py:extract_v2_metrics`): Reynolds (`Re = ρ·U_avg·D_h/µ`), Péclet streamwise/crossstream (`Pe = U·ℓ/D`), `aspect_ratio = W/H`, and `R²-to-linear` (least-squares fit of binned C against `a + b·ξ`, reported only for monotonic linear targets).
- **New hard BO constraints** (`ooc_optimizer/optimization/bo_loop.py`): `Re_max ≤ 100` (laminar gate), `aspect_ratio_max ≤ 15` (PDMS-collapse gate, user choice of 15 over the literature-conservative 10). The existing `tau_mean ∈ [0.1, 2.0]` and `f_dead ≤ 0.08` constraints are retained. The BO acquisition (ConstrainedExpectedImprovement over ModelListGP) auto-adapts to 5 constraint GPs.
- **Ladder topology integrated** (`ooc_optimizer/geometry/topology_blockmesh.py:_bm_ladder` lifted from the Phase-1 prototype; `ooc_optimizer/cfd/solver.py:_setup_case` dispatches to multi-inlet BC writers when `BlockMeshResult.inlet_names` is non-empty).
- **Target axis flipped to y** (`examples/tumor_chip_linear_gradient/config.yaml`).
- **Cross-topology comparison report extended** (`examples/tumor_chip_linear_gradient/run.py:_write_comparison_report`): per-topology S1/ST Sobol cross-tabulation, constraint-feasibility table, and per-topology median diagnostic metrics.

**Bug encountered + fixed mid-pipeline.** First production run crashed on `opposing` at the Sobol→BO handoff: `botorch.exceptions.errors.InputDataError: Input data contains NaN values.` Root cause: `PENALTY_METRICS["Re"] = float("nan")` in `solver.py` propagated through `c4 = Re_max - NaN`, poisoning the constraint GP fit. Fix: replaced NaN sentinels with finite "deeply infeasible" values (`Re=1e6`, `aspect_ratio=1e3`) in `solver.py` and `metrics.py` failure paths; defensive guard in `bo_loop._evaluate_point` updated to map NaN → finite-large (was NaN→`inf`, also unfittable). Bug + fix recorded in `tip.md`.

### 2.7 Extended H-sweep (completed)

Following the production integration, a focused **ladder-only test across H=200 and H=300 μm** was run to determine whether opening chamber height — which lifts the `aspect_ratio_max=15` cap from W ≤ 3000 to W ≤ 4500 μm — drops L2 below the production winner (0.0818). Config at `examples/tumor_chip_linear_gradient/config_ladder_H_sweep.yaml`; pre-sweep H=200 ladder artifacts archived to `_pre_H_sweep_20260426_142449/` to preserve the production winner reference.

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

### 3.4 4-topology integration BO campaign — actual results (2026-04-26)

200 evals/topology against `linear_gradient axis=y`, under the new 5-constraint feasibility set (`tau_mean ∈ [0.1, 2.0]`, `f_dead ≤ 0.08`, `Re ≤ 100`, `W/H ≤ 15`).

| Topology | Best feasible L2 | n_feasible / n_total | Failure rate |
|---|---|---|---|
| **`ladder`** (winner) | **0.0818** | 89 / 200 | 55.5% |
| `opposing` | 0.8822 | 122 / 200 | 39.0% |
| `same_side_Y` | 0.9937 | 127 / 200 | 36.5% |
| `asymmetric_lumen` | 1.0875 | 47 / 200 | 76.5% |

**Ladder beats the next-best topology by 10.8× and the prior axis=x `opposing` winner (L2 = 0.6343) by 7.76×.** The improvement is real — `R²_to_linear = 0.987` on the ladder winner confirms the field is linear-gradient-shaped to within 1.3% of variance.

**Ladder winner detail:**

| Field | Value | Note |
|---|---|---|
| L2_to_target | **0.0818** | 25% below the Phase-1 single-shot baseline (0.110) |
| R²_to_linear | **0.987** | the field is 98.7% linear in y |
| C_mean | 0.500 | symmetric by construction |
| W | 2999.6 μm | **constraint-active** at W/H ≤ 15 cap |
| Q_total | 119.5 μL/min | interior optimum |
| tau_mean | 1.992 Pa | **constraint-active** at upper bound 2.0 |
| f_dead | 0.031 | well within 0.08 cap |
| Re | 24.9 | far below `Re_max = 100` |

The optimum sits **on two constraints simultaneously** (`aspect_ratio_max` and `tau_mean_max`). The achievable L2 floor inside this constraint box is ~0.082; opening either constraint would drop it further.

**Cross-topology Sobol indices** (per `bo_<topology>_none_H200/interpretability/summary.json`, n_sobol=1024):

| Topology | Dominant param | S_T | Subdominant | S_T | ΣS_T | Trust |
|---|---|---|---|---|---|---|
| `ladder` | `Q_total` | 0.871 | `W` | 0.143 | 1.014 | ✓ |
| `asymmetric_lumen` | `r_flow` | 0.976 | `Q_total` | 0.033 | 1.022 | ✓ — surrogate is essentially 1-D in `r_flow` |
| `same_side_Y` | `r_flow` | 0.860 | `W` | 0.128 | 1.065 | ✓ |
| `opposing` | `delta_W` | 0.776 | `r_flow` | 0.651 | **1.813** | ⚠ overfit suspect (high failure rate compressed GP) |

The cross-topology contrast is the publishable design heuristic: **for prescribed-gradient targets, ladder is dominated by `Q_total` (a flow knob); two-inlet topologies are dominated by `r_flow` (an interface-position knob). These are fundamentally different control regimes.**

**Constraint-binding summary (H=200 ladder):**

| Constraint | Threshold | Observed | Verdict |
|---|---|---|---|
| `Re_max` | ≤ 100 | max Re = 24.98 across 800 evals | ✓ Never binds; safety rail. |
| `aspect_ratio_max` | ≤ 15 | binding (winner at 14.996) | ✓ Within expectation (literature 10–20). Most active manufacturability constraint. |
| `tau_mean_max` | ≤ 2.0 Pa | binding (winner at 1.998) | ⚠ At biology upper limit. Cell-line-specific (see §4). |
| `f_dead_max` | ≤ 0.08 | winner at 0.031 | ✓ Comfortable. Dominant infeasibility cause for `asymmetric_lumen` (117/200 fails) — lumen geometry creates stagnant corners. |
| `tau_mean_min` | ≥ 0.1 | rarely binds | ✓ |

**Local sensitivity at the H=200 ladder optimum** (`|∂μ/∂x_norm|`):

| Param | Local | Sobol S_T |
|---|---|---|
| `Q_total` | 0.0276 | 0.871 |
| `W` | 0.0149 | 0.143 |

Q dominates because it raises advective Pe (preserves stripes against transverse diffusion) AND raises shear (binds `tau_mean` against the cap). One knob doing two things.

**Fabrication-tolerance intervals at H=200** (10% L2 degradation budget):

| Param | Optimum | -Δ allowed | +Δ allowed | Tolerable range |
|---|---|---|---|---|
| `W` | 3000 μm | -1143 | +1501 | **[1857, 4500] μm** (~±40%) — far beyond PDMS soft-litho ±5–10 μm precision |
| `Q_total` | 119.5 μL/min | -42.8 | +80.2 | **[77, 200] μL/min** (~±40%) — far beyond syringe-pump ±1–2% precision |

The design is robust to fabrication and operational variance — does not require precision microfab.

### 3.5 Extended H-sweep — ladder at H=200 vs H=300 (axis=y, BO 200 evals each)

Wall time: **13 min 22 s**. 0 errors. Both runs at 200/200 evals.

| H (μm) | Best feasible L2 | Best W (μm) | Best Q (μL/min) | tau_mean (Pa) | aspect_ratio | f_dead | Re | R²-to-linear | C_std | Feasibility rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 (production) | 0.0817 | 2999 | 119.81 | **1.998** (cap) | **15.00** (cap) | 0.031 | 24.97 | 0.987 | 0.314 | 37.0% |
| **300 (winner)** | **0.0671** | **4496** | **200.00** (cap) | 1.483 | **14.99** (cap) | 0.021 | 41.71 | **0.990** | 0.306 | **96.5%** |

**Δ vs H=200**: L2 drops by **17.9% (0.0817 → 0.0671)** — matches the pre-test prediction (W≈4500, L2≈0.069) within 2%.

**Constraint-corner shift.** At H=200 the optimum was **double-bound on `aspect_ratio_max` AND `tau_mean_max`**. At H=300 the corner has moved: `aspect_ratio_max` still binds (W = 4496 ≈ 4500), but `tau_mean` is no longer binding (1.48 ≪ 2.0); instead the **`Q_total` upper bound (200 μL/min) is now binding**. The tau-relief is the predicted physics consequence of doubling-and-a-half H: at fixed (W, Q), `τ ∝ Q / (W·H²)`, so the ~2.25× reduction in τ-density (H ratio squared) more than compensates the 1.67× increase in Q at the new optimum.

**Feasibility nearly tripled** (37% → 96.5%) because the wider H pushes the tau-feasibility region to encompass most of the search box. This is a major collateral benefit beyond L2 — at H=300 the BO wastes far fewer evals on infeasible corners.

**Sobol indices** (per H, n_sobol=1024):

| H | Q_total S_T | W S_T | ΣS_T | Order change |
|---|---|---|---|---|
| 200 | 0.871 | 0.143 | 1.014 | — |
| 300 | 0.861 | 0.152 | 1.013 | none — `Q_total` still dominant; `W` slightly more influential at H=300 (since W is now the only freely-binding constraint) |

**Tolerance intervals at H=300** become asymmetric — both `W` and `Q_total` are pinned at their upper bounds, so the +Δ direction has nearly zero room (`W: +4.4 μm`, `Q_total: +0`) while the −Δ direction has substantial slack (`W: -1675 μm`, `Q_total: -72.5 μL/min`). The optimum sits in a **corner** of the design box, with all the headroom on the "smaller" side. This is what a well-converged constrained BO looks like.

**Six interpretation points worth highlighting in the writeup:**

1. **The 17.9% L2 drop matches the pre-test prediction within 2%.** Production-run interior data showed unconstrained candidates near W=4400, Q=180 reaching L2≈0.069; the H=300 BO converged to W=4496, Q=200, L2=0.0671. This is not luck — it is what the constraint-relaxation analysis told us would happen.
2. **The constraint corner moved rather than opened.** AR still binds (W=4496 ≈ 4500 cap); `tau_mean` releases (1.48 Pa, well below 2.0); `Q_total` upper bound is now binding. L2 = 0.0671 is the floor inside this *new* corner, not the topology's intrinsic floor.
3. **The dimensional-analysis prediction `τ ∝ Q/(W·H²)` matches observation to within 1.5×.** Predicted tau at H=300, W=4500, Q=200: ~0.99 Pa from a fully-developed 6μU/H argument; observed: 1.48 Pa. The gap is the inlet-region acceleration that the textbook formula ignores. Order-of-magnitude correct.
4. **Feasibility leapt from 37% → 96.5%** because the larger H pushes the tau-feasibility region to encompass most of the search box. Same compute, **2.6× more useful evaluations** — the H=300 GP surrogate is much higher-quality than the H=200 surrogate.
5. **Sobol order unchanged (`Q_total` dominant at S_T ≈ 0.86 at both H), but `W`'s share rises slightly** (0.143 → 0.152) because at H=200 the AR cap pinned W so tightly that BO had no real search room. ΣS_T ≈ 1.01 at both H — clean, trustworthy indices.
6. **R² climbs from 0.987 → 0.990** (residual non-linearity halved, 1.3% → 1.0%). C_std drops from 0.314 → 0.306 toward the theoretical max 0.289. The H=300 field is *more linear and less dynamic-range-overshooting* — diffusive smoothing at the larger H smooths the inlet ladder steps without erasing the gradient.

**Three things the H-sweep cannot answer (open questions for follow-up):**

- The intrinsic ladder floor below 0.067 is unknown — `Q_total = 200 μL/min` is a YAML choice, not a physical limit (typical syringe-pump max ~1000). #2 in §6 addresses this with a one-line YAML change.
- H-sensitivity is sampled at only two points (200, 300). H=250 might be a Pareto sweet spot for imaging applications where smaller H aids cell visualisation. A finer H grid is a separate diagnostic, not a BO campaign.
- Per-inlet `C_k` headroom is unmeasured. The R²=0.990 ceiling at H=300 is set by within-strip step quantisation plus small numerical diffusion. Only the per-inlet `C_k` 8-D BO (#1 in §6) can quantify how much of the residual 1% R² is recoverable.

Full per-H analysis at `examples/tumor_chip_linear_gradient/data/results/ladder_H_sweep_findings.md`. Full integration-run analysis at `examples/tumor_chip_linear_gradient/data/results/integration_run_findings.md`.

---

## 4. Interpretation and Discussion

The diagnostic phase produced a clean, mechanistic answer that the BO results alone could not have settled: **the L2 ≈ 0.585 floor on the original three topologies is geometric, not algorithmic.** No length of BO campaign, no widening of bounds, and no re-tuning of acquisition function could have broken it. The two-inlet coflow geometry, in the laminar regime mandated by the cell-biology constraints, can only produce a transverse step — and the original target was specified along x. Mass conservation is the invariant that makes this conclusion robust.

### 4.1 The topology-vs-optimization L2 stack

Topology selection does most of the heavy lifting; geometry/flow optimization adds modest but cumulative gains; and each layer earns its keep:

| Stage | Best L2 | Multiplicative gain |
|---|---|---|
| Original 3 topologies + axis=x BO | 0.6343 | 1.0× (baseline) |
| Topology pivot to ladder + axis=y, hand-picked (W=3000, Q=50, midpoint C_k) | 0.110 | **5.8×** ← topology contribution |
| H=200 ladder + 2-D BO over (W, Q_total) | 0.0818 | × 1.34 (25% over hand-picked) |
| H=300 ladder + 2-D BO + relaxed AR cap | **0.0671** | × 1.22 (17.9% over H=200) |
| (Hypothetical) per-inlet `C_k` 8-D BO + mesh refinement | ~0.04 | × ~1.7 (estimated) |

**Topology buys the largest single jump (≈6×); each subsequent BO/constraint-relaxation layer adds ~1.2–1.3×.** Equally important, the BO and Sobol layers produce the **fabrication-tolerance intervals, dominant-parameter ranking, and constraint-binding diagnostics** that no hand-picked design could deliver. Even when the absolute L2 gain is modest, those *interpretability artifacts are the project's publishable methodology contribution*.

### 4.2 What the ladder winner tells us — the constraint-corner shift

The H=200 ladder optimum sat at a **double-bound corner**: `aspect_ratio_max=15` and `tau_mean_max=2.0` both binding within 0.1%. Top-10 candidates clustered tightly (W ∈ [2987, 3000], Q ∈ [117.9, 119.5], L2 ∈ [0.0818, 0.0820]) — a sharply localised but degeneracy-tolerant optimum. P10 of feasibles = 0.0820, P90 = 0.0857; the L2 surface in the feasible region is essentially flat at 0.082 ± 1%.

Opening H to 300 μm shifted the corner: AR still binds (W = 4496 ≈ 4500), but `tau_mean` falls to 1.48 (no longer binding) and **`Q_total` saturates at its 200 μL/min upper bound**. Feasibility leapt from 37% to 96.5%, and L2 dropped 17.9%. The tau-relief is exactly what the dimensional analysis predicts: τ ∝ Q/(W·H²), so doubling-and-a-half H gives ~2.25× tau headroom that the BO immediately spends on higher Q for sharper gradients. **The ladder L2 floor at AR=15 is now ≈0.067**, with the limiting bottlenecks shifted from "shear + chamber width" to "chamber width + flow rate".

The corner shift has a concrete next-step implication: **further L2 improvement at fixed AR=15 will not come from W or Q_total tuning** — both are pinned at upper bounds at H=300. The remaining levers are (i) per-inlet `C_k` (BC fidelity), (ii) mesh refinement (numerical-diffusion floor), or (iii) opening the parameter box (`Q_max → 400`, `H = 400`).

### 4.3 Cross-topology Sobol — the publishable design heuristic

For prescribed-gradient targets, the dominant control variable changes with topology:

- **Ladder**: `Q_total` carries 87% of S_T because it does double duty (raises advective Pe AND raises shear); `W` is secondary (15%). No parameter interaction (ΣS₁ ≈ 0.99, ΣS_T ≈ 1.02).
- **Two-inlet topologies (`same_side_Y`, `asymmetric_lumen`)**: `r_flow` dominates (86–98% S_T) because it is the only knob that moves the y-position of the drug/medium interface — the only structural feature these topologies can produce. `asymmetric_lumen` is essentially a 1-D problem in `r_flow`.
- **`opposing` Sobol indices are not trustworthy** (ΣS_T = 1.81). High failure rate (39%) compressed the GP into near-interpolation despite the noise floor we previously added. Treat its `delta_W` dominance claim with caveat — directionally consistent with prior diagnostics but magnitudes inflated.

This contrast — *flow-dominated control for ladder vs. interface-position-dominated control for two-inlet topologies* — is a clean, defensible design heuristic that no published paper has reported for tumor-on-chip gradient chambers (per the literature survey).

### 4.4 Constraint and manufacturability assessment

**`Re_max ≤ 100` works as a safety rail**: max Re across 1200 evals (800 first run + 400 H sweep) was 41.71 (the H=300 ladder winner). Never binds. The choice to add this constraint was correct in principle (would catch low-Q-and-tight-chamber corners that approach turbulent regime) but didn't earn its keep numerically — could be removed in future minimal configs.

**`aspect_ratio_max = 15` is the most active manufacturability constraint**, binding at every L2-best record across both H levels. We chose 15 over the literature-canonical 10 to give the BO headroom. The trade-off: at W/H = 15 we are at the *edge* of the safe PDMS-bonding range (literature consensus 10–20). For chip-design demonstration this is fine; for **multi-replica fabrication** consider tightening to 12 or 10 (cost ~3% L2). At H=300, AR=15 corresponds to W=4500 μm — still routine for soft-litho.

**`tau_mean_max = 2.0 Pa` binds at H=200 but not H=300**, and the 2.0 limit itself is **cell-line-specific**:

- **Tolerant lines** (immortalised lines, kidney/hepatocyte lines, endothelial cells) sustain 2–5 Pa for hours — both H winners are in their safe range.
- **Sensitive lines** (primary tumor cells, neurons, weakened-cytoskeleton variants) stress at 1–2 Pa over multi-hour experiments. The H=200 winner (1.998 Pa sustained) is uncomfortable for these. The H=300 winner (1.48 Pa) is comfortable for most cell lines.

This is a genuine biological recommendation: **specify the cell line, then re-evaluate the tau cap.** If the experiment uses sensitive cells, drop `tau_mean_max` to 1.0 Pa. The H=300 winner already lies inside that tighter window, while the H=200 winner does not — another reason to prefer H=300 even before the L2 improvement is counted.

**`f_dead_max ≤ 0.08`** is comfortable for ladder (0.021–0.031) but is the dominant infeasibility cause for `asymmetric_lumen` (117/200 fails) — the lumen geometry creates stagnant corners that the BO cannot reduce below the threshold. This is a topology-intrinsic limitation.

### 4.5 Honest caveats

- **`opposing` Sobol indices have ΣS_T = 1.81**, above the 1.5 trustworthy threshold. Surrogate is overfit; report the magnitudes with caveat.
- **Two non-fatal `Field T uniform 0` extraction errors** appeared across the integration runs. Frequency ~1%, treated as penalty L2=99, no bias on the BO.
- **Phase-2 (W, Q_total) Sobol scan from earlier diagnostic was *flat* (L2 ∈ [0.172, 0.176])** but the production BO winner is at 0.082 (H=200) / 0.067 (H=300). The discrepancy is explained: Phase-2 used the *endpoint* convention `C_k = k/(N-1)` (which gives 38% higher L2 at any fixed geometry); production uses the *midpoint* convention `(k+0.5)/N`. The midpoint convention is now the production default and is a free 38% improvement.
- **The H=300 winner has Q_total pinned at the 200 μL/min upper bound.** This means the actual achievable L2 floor for ladder under all 5 constraints is somewhere below 0.067; we cannot quantify it without raising the Q upper bound. Worth noting in the writeup that 200 μL/min is a YAML choice, not a hard physical limit (typical syringe-pump max is ~1000 μL/min).
- **Topologies B–E (christmas_tree, side_injection, permeable_wall, counter_flow)** were screened with first-principles physical analysis but not implemented. Each has a "hidden parameter" set whose value determines whether the C-field is actually linear (vs. plateau, vs. saturating, vs. step-with-impingement-noise). Each requires non-trivial mesh and BC engineering; on the roadmap, not abandoned.

### 4.6 Novelty framing

The literature survey (Yang 2020 RSC Adv 10:13799, Hashemi-Tilehnoee 2025 Lab Chip, Whitesides 2000, Dertinger 2001, Ayuso 2020 IJMS 21:9075, Borrvall–Petersson 2003) confirmed that the ladder + Kriging/BO surrogate design is published prior art for general microfluidic mixers. **The combination of (i) tumor-on-chip 3D-culture chamber, (ii) cross-topology Sobol sensitivity comparison on CFD-trained GP surrogates, (iii) explicit fabrication-tolerance reporting derived from the surrogate, and (iv) the dimensional-physics constraint set (Re, aspect ratio, tau, f_dead) wired into the BO acquisition is novel.** That bundle is the project's defensible methodology contribution; the BO and ladder topology are well-engineered scaffolding around it.

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

### 5.5 Integration-run artifacts (axis=y, completed)
- `examples/tumor_chip_linear_gradient/data/results/_aborted_run_axisX/` — pre-axis-flip campaign
- `examples/tumor_chip_linear_gradient/data/results/_aborted_run_axisX_opposing_only/` — partial pre-fix opposing crash
- `examples/tumor_chip_linear_gradient/data/results/evaluations_<topology>_none_H200.jsonl` — 4 topologies' axis=y eval logs (200 each), with the H=200 ladder log being **post-H-sweep** (replaces the production winner reference, archived in §5.6)
- `examples/tumor_chip_linear_gradient/data/results/evaluations_ladder_none_H300.jsonl` — H=300 ladder eval log (200 evals)
- `examples/tumor_chip_linear_gradient/data/results/optimization_summary_opposing.json` — the recovered `opposing` rerun summary
- `examples/tumor_chip_linear_gradient/data/results/optimization_summary_ladder_H_sweep.json` — H-sweep summary (winner: ladder_none_H300, L2=0.0671)
- `examples/tumor_chip_linear_gradient/data/results/bo_<topology>_none_H<H>/` — 5 BO state directories: opposing/same_side_Y/asymmetric_lumen/ladder at H=200, plus ladder at H=300
- `examples/tumor_chip_linear_gradient/data/results/bo_<topology>_none_H<H>/interpretability/{summary.json, sobol.png, local_sensitivity.png, tolerance.png, design_heuristics.md}` — per-topology Sobol artifacts
- `examples/tumor_chip_linear_gradient/data/results/integration_run_findings.md` — comprehensive 4-topology integration analysis (Sobol cross-tab, constraint binding, axis-flip diagnostic)
- `examples/tumor_chip_linear_gradient/data/results/ladder_H_sweep_findings.md` — focused H=200 vs H=300 comparison

### 5.6 Pre-H-sweep ladder reference (archived)
- `examples/tumor_chip_linear_gradient/data/results/_pre_H_sweep_20260426_142449/` — production H=200 ladder run (L2=0.0818) preserved for direct comparison against the H-sweep H=200 redo

### 5.7 Source / scripts (for the methodology section of a writeup)
- `scripts/visualize_topology_candidates.py` — generates §5.1
- `scripts/diagnostic_baseline.py` — closed-form uniform-field check
- `scripts/diagnostic_ladder_baseline.py` — Phase-1 single-shot
- `scripts/diagnostic_ladder_scan.py` — Phase-2 Sobol scan
- `scripts/run_optimization.py` — production BO entry
- `scripts/run_interpretability.py` — Sobol on GP surrogate
- `examples/tumor_chip_linear_gradient/run.py` — comparison-report writer

### 5.8 Project documentation
- `tip.md` — bug history (case-dir collision fix; numerical diffusion notes)
- `README.md` — revision history (will receive a one-line entry on integration completion)
- `Development_Guide_v2.md` — architectural reference; the BO + interpretability framing in §1 of this report follows it

---

## 6. What's next

The current state is a clean stopping point for a publication or poster. Recommended further steps, ordered by cost/benefit:

1. **Per-inlet `C_k` 8-D BO at fixed (W=4500, H=300, Q=200)** — the H=300 winner has W and Q both pinned at upper bounds, so geometry/flow knobs are exhausted; the only remaining lever is **boundary-condition fidelity**. Letting the BO choose 8 monotonic `C_k ∈ [0,1]` values (instead of the fixed midpoint convention) lets it pre-compensate for cross-stream diffusion that smears the imposed inlet ladder. Implementation: ~1 afternoon (extend `PARAMETER_ORDER` for ladder, mask out the existing 7 dims, add 8 active C_k dims). Expected L2 floor: ~0.04 (analytical within-strip residual + small numerical-diffusion residual).
2. **Open the search box** — raise `Q_total_max` from 200 → 400 μL/min (typical syringe-pump capacity is ~1000) and add H=400 as a discrete level. This pushes the AR=15 cap to W ≤ 6000 μm and gives BO more genuine optimisation room. YAML-only changes. Should be done before #1 to confirm whether L2 < 0.067 is tractable just by box-opening.
3. **Mesh refinement** (`ny_per_mm: 25 → 60`, sharper advection scheme) — only after #1 + #2; drops the numerical-diffusion contribution to L2 by ~50%. 3× per-eval wall time, so reserve for a final "publication-quality" rerun on the established winner.
4. **Cell-line-specific reruns** — if the experimental partner uses sensitive cells, rerun ladder with `tau_mean_max = 1.0 Pa`. The H=300 winner already lies inside that tighter window; H=200 does not. Worth documenting as a "biology-aware" variant.
5. **Topologies B–E** — implementation roadmap unchanged; `side_injection` (1 day) is the only path to a true axis=x linear gradient if a follow-up project ever needs it (e.g. spheroid-PK studies); `christmas_tree` (1.5 days) is the fab-realistic version of the ladder for chips that have only 2 reagent reservoirs.
6. **Density-based topology optimization** (Borrvall–Petersson 2003 in Brinkman-penalised Navier–Stokes) — separate research project, publishable as a standalone paper. Explicit roadmap in `diagnostic_findings.md`. Out of scope for the current cycle.

The "minimum useful next move" is **#2 then #1** (~1 day of work total), which would push L2 to ~0.04 and give a final tolerance/sensitivity report on a single optimal design — appropriate for the manuscript.
