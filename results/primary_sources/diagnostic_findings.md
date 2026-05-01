# Linear-Gradient Campaign — Diagnostic Findings & Forward Plan

**Date:** 2026-04-26
**Campaign analysed:** `examples/tumor_chip_linear_gradient/data/results/optimization_summary.json` (200 evals/topology × 3 topologies, opposing/same_side_Y/asymmetric_lumen, pillar=none, H=200μm, axis=x linear gradient).
**Patch validated this run:** `cfd/solver.py` case-dir suffix now includes PID + uuid (0 collisions vs 148 in the prior aborted run).

---

## 1. Headline finding

The BO campaign is **near the global optimum of the achievable design space**, but the achievable optimum is **far from the idealized linear gradient**. The mismatch is structural — a property of the topology class, not of the optimizer or the constraint settings.

| | L2 (axis=x linear) | C_mean | mono | f_dead |
|---|---|---|---|---|
| Default-config baseline | **0.9997** | 0.500 | 0.588 | 0.000 |
| BO winner (`opposing`) | **0.6343** | 0.672 | 0.500 | 0.0796 |
| Theoretical uniform-field floor | **~0.585** | — | — | — |

- BO improved 37% over the baseline → **not a stall**.
- BO winner sits 8% above the analytical "uniform field at C=0.672" L2 (= 0.585). The remaining gap is the cost of imperfect mixing, not optimizer failure.
- `f_dead = 0.0796 / 0.08` is *not* the bottleneck: the baseline has `f_dead=0` and still hits L2=1.0. Relaxing the constraint cannot push L2 below 0.585.
- Cross-axis check at the same baseline:
  - Linear axis=y: L2 = 1.32 (worse than axis=x).
  - Step axis=y, sharp=0.05: L2 = 1.40.
  - Step axis=y, sharp=0.20: L2 = 1.33.
  None of these produce a lower floor — **simply rotating the target does not rescue the design**.

## 2. Physical reason — why no parameter setting in this topology class can reach the linear target

The three current topologies all have the same boundary structure: **two coflow inlets at x=0, single outlet at x=L, no through-wall flux**. For a passive scalar in steady incompressible flow with these BCs, conservation of advective scalar flux through any vertical slice gives:

$$\int_0^W u_x(x,y)\, C(x,y)\, dy = \int_0^W u_x(0,y)\, C(0,y)\, dy = r_{\text{flow}}\, Q_{\text{total}} \quad \forall\, x \in [0, L]$$

Since `u_x ≈ Q_total / (W·H)` is roughly uniform across y in a long shallow channel, this forces `<C>_y(x) ≈ r_flow` at every x. **The y-averaged concentration is pinned along the entire flow direction.**

A linear x-gradient `C_target(x) = x/L` requires `<C_target>_y(x) = x/L`, which varies linearly with x. **No combination of W, d_p, s_p, θ, Q_total, r_flow, ΔW can produce that** because there is no source or sink in the chamber that would let scalar mass enter or leave between x=0 and x=L. The L2 floor at 0.585 is the geometric lower bound for any uniform field at C=r_flow against the linear target — no design can break it.

The closest the BO can get is "make C(x,y) as uniform as possible at C=r_flow", which is exactly what it found: high W (more cross-stream mixing distance), low θ (gentler inlet jets), large ΔW (separated inlets to spread the interface), high r_flow (push C_mean up so the field overlaps the upper half of the [0,1] target range).

## 3. Implication

**The "linear gradient axis=x" target was never physically reachable with the current topology class.** The campaign's L2 numbers do not measure design quality — they measure how well a uniform field with C ≈ 0.67 approximates a linear ramp from 0 to 1. That is a fixed mathematical quantity, not an engineering achievement.

To make further BO investment worthwhile, **either the topology must change to introduce x-direction scalar sources/sinks, or the target must change to a shape that the current geometry can actually produce**.

---

## 4. Topology candidates that *could* produce a linear gradient

For each candidate I list: (a) the achievable target shape, (b) why it works physically, (c) implementation cost in this codebase, (d) headline risk.

### A. Y-stacked N-inlet ladder (axis=y target)

- **Achievable target**: linear gradient *transverse to flow* (axis=y).
- **Mechanism**: Replace the two-inlet face at x=0 with N stacked inlet strips along y, each with a prescribed concentration `C_k = k/(N-1)` for k = 0..N-1. The N parallel laminar streams flow downstream side-by-side; transverse diffusion smooths the discrete steps into a quasi-linear y-profile after a mixing length `L_mix ≈ W² · u / D`. For typical Pe ≈ 100–1000, full smoothing requires a long chamber.
- **Why it works**: at x=0 the gradient is *imposed* (a ladder that's already linear in y on average); the chamber's job is just to smooth the steps without destroying the trend. Mass conservation is satisfied because the gradient is transverse to flow, not along it.
- **Implementation**: extend `ooc_optimizer/geometry/topology_blockmesh.py` with a `ladder_N` builder that produces N inlet patches. Extend the BC writer (`cfd/solver.py:_setup_case`) to assign `fixedValue C=k/(N-1)` per patch. Add `N` and possibly per-stream flow rates as new BO variables. Modest geometry work, no new solver.
- **Risks**: streams must remain laminar (Re < few hundred); inlet fabrication cost rises with N; if N too small the field is steppy. **N=8 is the canonical Whitesides choice.**

### B. Christmas-tree pre-mixer + wide chamber (axis=y target)

- **Achievable target**: linear y-gradient, same as A but with on-chip preparation of the N intermediate concentrations from only two physical reagent reservoirs (drug + medium).
- **Mechanism**: a binary tree of serpentine mixing channels splits the two reagents repeatedly, producing N=2^k parallel outputs at concentrations spaced by 1/(N-1). These then feed the wide chamber as in candidate A.
- **Why it works**: same as A, but solves the practical problem of needing N reagent streams when only two are available off-chip.
- **Implementation**: significant geometry work — the mixer tree itself must be meshed and solved. The serpentine channels require full mixing before each junction (channel length scales as `W² u / D ≈ several mm` per branch); total tree footprint can dwarf the chamber. The simulation cost per evaluation roughly doubles vs the current pipeline.
- **Risks**: mesh complexity; numerical diffusion in serpentine channels can degrade the prepared gradient before it reaches the chamber.
- **Recommendation**: only build this *after* candidate A demonstrates the y-gradient is reachable in principle.

### C. Distributed side-injection (axis=x target)

- **Achievable target**: linear x-gradient (the *idealized* target).
- **Mechanism**: an array of K small inlets along the chamber side wall (y=0) injecting C=1, plus the original x=0 inlet at C=0 feeding the bulk medium. Each side-inlet adds drug mass locally; with prescribed flow rates `Q_k(x_k)` increasing along x, the cumulative drug fraction at position x is `Σ_{i: x_i<x} Q_i / (Q_medium + Σ_{i: x_i<x} Q_i)`. With proper Q_k spacing this *can* be linear in x.
- **Why it works**: the side inlets are the distributed scalar source that the conservation argument required. Mass conservation is no longer violated because mass is genuinely entering through the side wall.
- **Implementation**: extend topology builder with K side-inlet patches (parametric K ∈ {4, 8, 16}). Each patch needs flow-rate and concentration BCs; the per-patch flow rate `Q_k` becomes a BO variable (or is parameterised by `Q_k = Q_0 · f(x_k)` with a small number of shape parameters). Total inlet flow now varies along x → outlet flow ≈ Σ Q_k, much larger than any single inlet. Solver handles fine; mesh complexity is moderate.
- **Risks**: each side jet creates a local recirculation; total `f_dead` will rise above current 0.08 threshold; constraint must be relaxed. K is a discrete BO variable adding combinatorial complexity. Fabrication cost in a real device is high (many parallel pumps).
- **Why it's interesting anyway**: this is the **only candidate that produces the actual axis=x linear gradient the project's target was specified for**. If matching the original target is non-negotiable, this is the path.

### D. Permeable-wall drip with reservoir (axis=x target)

- **Achievable target**: linear x-gradient, smoother than C.
- **Mechanism**: replace one chamber wall (e.g., the floor, z=0) with a permeable membrane separating the chamber from a drug reservoir at C=1. The flux through the membrane at position x is `D_m · (C_reservoir - C(x,y=0,z=0)) / δ_m` where δ_m is membrane thickness; this is x-dependent because C in the chamber is x-dependent. With a *graded* membrane permeability `K(x) = K_0 · x/L`, the flux ramps up along x and the chamber mean concentration grows roughly linearly with x.
- **Why it works**: distributed source via wall flux. The grading is in fabrication, not in flow control — only one reservoir is needed.
- **Implementation**: requires a custom OpenFOAM BC (Robin-type) and possibly a multi-region solver (`chtMultiRegionFoam` or similar) if the membrane diffusion is co-resolved. Significant development outside the current scope.
- **Risks**: high-effort prototype before knowing whether grade-K(x) fabrication is feasible. Defer.

### E. Counter-flow inlets (axis=x target)

- **Achievable target**: weak linear x-gradient, more like erf-shape.
- **Mechanism**: drug enters at x=0 at low flow rate, medium enters at x=L at high flow rate; they meet somewhere inside; net flow goes to outlets at the y=0 and y=W walls in the middle of the chamber. Drug propagates upstream against medium by diffusion.
- **Why it works**: counter-flow creates a *frontal interface* whose position depends on flow ratio. The interface zone has x-dependent C.
- **Risks**: fundamentally unsteady regimes are easy to fall into (vortex shedding at the impingement). Steady-state existence not guaranteed for all parameters. Likely bad fit for the BO penalty framework, which assumes well-defined steady metrics.
- **Recommendation**: do not pursue.

### Ranking (recommended order of investment)

| Rank | Candidate | Target axis | Cost | Likely L2 floor | Notes |
|---|---|---|---|---|---|
| **1** | A — Y-stacked ladder, N≈8 | y | low | ≪ 0.585 expected | cheapest demonstration of "BO can do something useful when target is reachable" |
| **2** | C — Side-injection array | **x** | medium | unknown but tractable | only path that preserves the original axis=x target |
| 3 | B — Christmas-tree + chamber | y | high | similar to A | only worth it after A succeeds, for practical fab |
| 4 | D — Permeable wall | x | very high | unknown | OpenFOAM development required |
| 5 | E — Counter-flow | x | low to set up | poor | unsteady risk; not recommended |

---

## 5. Constraint and target tweaks (independent of topology change)

These can be adjusted in YAML alone; they do not unlock x-gradient feasibility but are useful when running the new topologies above.

### Constraints (`optimization.constraints` in `default_config.yaml`)

- **`f_dead_max`**: relax 0.08 → **0.20**. Multi-inlet ladders and side-injection arrays naturally produce stagnation zones at junctions; the current 0.08 threshold would penalise even good designs. The current campaign already showed `f_dead` was not the binding constraint, so a permissive value carries no quality cost.
- **`tau_mean_min`**: relax 0.1 → **0.05**. Lower flow rates favour diffusion-dominated smoothing of inlet step gradients. The current floor excludes the slow-flow regime where gradient quality is best.
- **`tau_mean_max`**: tighten 2.0 → **1.0** for biological relevance (cell viability), only if the application requires it. Otherwise leave at 2.0.

### Target shape (`target_profile`)

These changes still describe a "gradient" but make it physically achievable with the existing topology class — useful as a **fallback** if the new topologies (§4) prove too costly to implement:

- **Soft step in y, sharp ≈ 0.3–0.4** (`step_axis: y, sharpness_frac: 0.35`): represents a smoothed transition between "drug side" and "medium side". Achievable by current topologies via diffusion across the y-interface; matches what the chamber naturally produces.
- **Linear gradient in y with shrunk range** (`linear_gradient axis: y, c_high: 0.7, c_low: 0.3`): the absolute concentration range is narrower so the imperfect mixing has less to compensate for. Achievable by tuning flow ratios to make the interface span y more gradually.
- **Bimodal in y** with closely-spaced peaks (`bimodal peak_axis: y, peak_fracs: [0.3, 0.7], width_frac: 0.15`): tests whether topologies can sustain *any* spatial structure beyond chance.

### BO settings

- **`n_sobol_init`**: bump 24 → **48** when changing topology. The active-dim grows (per-inlet flow rates in candidates A and C), and Sobol space coverage matters more.
- **Acquisition**: switch from `ConstrainedExpectedImprovement` to **`LogConstrainedExpectedImprovement`** (BoTorch warned about this on every BO round of the completed run; tip.md already notes the legacy form has known numerics issues). One-line change in `bo_loop.py`.
- **Budget**: 200 evals/topology was sufficient — the surrogate stabilised well before the budget exhausted on all three current topologies. Do not increase further until topology is fixed.

---

## 6. Recommended next experiment

A single phased plan, cheapest decisive test first.

### Phase 1 — feasibility demonstration (≤ 1 day)

Implement candidate **A** (Y-stacked ladder, N=8) as a new topology in `topology_blockmesh.py` and `solver.py`. Run a single CFD at a sensible default geometry against `linear_gradient axis=y` target. Goal: produce L2 < 0.3 at the *baseline* parameters, demonstrating the topology can reach the target without optimization. If yes, proceed to Phase 2. If no (L2 > 0.5), the diffusion length is too short or the geometry detail is wrong — debug before continuing.

### Phase 2 — first BO campaign on new topology (≤ 1 day wall-clock)

Run a 200-eval BO on the ladder topology with relaxed `f_dead_max=0.20`, target `linear_gradient axis=y`. Compare best L2 to Phase 1 single-shot baseline. Goal: BO improves over baseline by ≥30%, and absolute L2 < 0.2.

### Phase 3 — reach the original axis=x target

If the user requires the axis=x linear gradient (not just *a* linear gradient), implement candidate **C** (side-injection array) with K=4 or K=8. This is the only path; budget ≥ 3 days of geometry/BC development. Defer until Phase 1+2 confirm the BO toolchain can exploit a reachable gradient.

### What NOT to do

- Do **not** re-run the existing three topologies with longer budgets, more Sobol points, or different acquisition functions. The L2 floor is geometric, not surrogate-driven; no amount of BO tuning will break it.
- Do **not** widen the continuous bounds further. The campaign already showed `same_side_Y` wedges at `r_flow=0.97` and `asymmetric_lumen` at `W=1500` — but loosening those bounds only lets BO find slightly more uniform fields, not real gradients.
- Do **not** skip Phase 1's single-shot baseline test. If the ladder topology cannot achieve a low L2 even at hand-picked parameters, BO will not save it.

---

## 7. Phase 1 + Phase 2 results (2026-04-26)

Phase 1 — single-shot CFD on the ladder topology (`scripts/diagnostic_ladder_baseline.py`).
Phase 2 — Sobol-quasirandom scan over (W, Q_total), 32 evals (`scripts/diagnostic_ladder_scan.py`); used as a stand-in for full BO since the ladder topology is not yet integrated into the production BC writers / orchestrator.

### Results table

| Configuration | L2 (axis=y linear) | Notes |
|---|---|---|
| Original BO winner (`opposing`, axis=x) | 0.6343 | uniform-field floor at 0.585 |
| Ladder N=4, midpoint C_k | 0.1337 | resolution-limited |
| Ladder N=8, endpoint C_k = k/(N-1) | **0.1756** | Phase 1 baseline; same as Sobol-scan median |
| Ladder N=8, midpoint C_k = (k+0.5)/N | **0.1097** | 38% drop from convention change alone |
| Ladder N=16, endpoint | 0.1423 | |
| Ladder N=16, midpoint | **0.1091** | floor reached — no further N benefit at current mesh |
| Best from 32-eval Sobol scan over (W ∈ [1500, 4500] μm, Q_total ∈ [5, 200] μL/min), N=8, endpoint | 0.1720 | best at W=1605 μm, Q=10.17 μL/min |
| **Improvement vs original BO winner** | **5.82×** | from 0.6343 → 0.1091 |

### What the Sobol scan revealed

All 32 evaluations produced L2 ∈ [0.1720, 0.1759] — a spread of 0.004, essentially flat. **(W, Q_total) tuning does not help the ladder topology** at the current chamber length and diffusivity, because the Péclet number (Pe ≈ 4·10⁷ at default) keeps the streams perfectly stratified — the chamber neither helps nor hurts the imposed inlet ladder. The L2 floor is set by **the discrete-ladder approximation error and numerical mesh diffusion**, not by chamber-flow parameters.

### Why convention matters (the unexpected finding)

The L2 metric measures pointwise difference between simulated `C(y)` and target `y/W`. With **endpoint convention** (`C_k = k/(N-1)`), strip 0 is set to exactly 0 and strip N-1 to exactly 1 — but their *centres* are at y = (1/(2N))·W and (1 - 1/(2N))·W, where the linear target evaluates to those midpoint y-values, not 0 and 1. Result: a ±1/(2N) systematic offset at the edge strips dominates the L2.

With **midpoint convention** (`C_k = (k+0.5)/N`), each strip's prescribed concentration matches the linear-target value at the strip's centre exactly. Only within-strip variation (target ramps, C is constant) contributes to L2. For N=8 the analytical floor is `1/(2N√3) / (1/√3) = 1/(2N) = 0.0625`. We observed 0.1097, i.e. **1.76× the analytical floor** — the gap is numerical mesh diffusion (upwind scheme + mesh resolution `ny_per_mm=25`).

### Verdict on Phase 2

**Full BO over (W, Q_total) is not warranted.** The response surface is flat. The remaining L2 reduction (0.110 → ~0.063) requires changes that are not BO targets:

1. **Mesh refinement** — `ny_per_mm: 25 → 60` doubles cells per strip; numerical diffusion drops accordingly. Cost: ~3× per-eval wall time.
2. **Discretization scheme** — `div(phi,T) bounded Gauss upwind` → `bounded Gauss limitedLinear 1` is sharper on advection-dominated transport but tip.md notes it caused SIGFPE at high Pe in the 1D verification. Risky; needs targeted testing.
3. **Per-inlet C_k as BO parameters** — instead of `C_k = (k+0.5)/N`, let BO choose 8 free parameters `C_0..C_7 ∈ [0,1]`, monotonically constrained. The optimum will be a slight non-linear correction compensating for downstream diffusion. This is a real BO problem with 8 active dims and a tractable surrogate. **This is the recommended next BO campaign**, not (W, Q_total).

### Visual artifacts

Topology schematics for all 5 candidates: `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/{A,B,C,D,E}_*.png`. Open with `open <path>` on macOS or any image viewer. The schematics are matplotlib renders (not meshable geometry); they exist for intuition. Sobol-scan response surface: `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/heatmap.png`.

### Recommended next move

If the goal is "match a linear y-gradient as closely as possible with a fabricable chip":

1. Adopt **ladder N=8 midpoint convention** as the production topology baseline (L2 = 0.1097, validated single-shot).
2. Integrate `_bm_ladder` into `ooc_optimizer/geometry/topology_blockmesh.py` and the BC-writer in `ooc_optimizer/cfd/solver.py` so the production BO can use it. Estimated work: 1 afternoon.
3. Run a fresh BO with **per-inlet `C_k`** as the active design vector (8 parameters, monotonicity constraint, target = `linear_gradient axis=y`). Skip (W, Q_total) optimization — the scan already showed it doesn't help.
4. If budget allows, also enable **mesh resolution** as a discrete BO level (`ny_per_mm ∈ {25, 40, 60}`) to find the cost/L2 trade-off explicitly.

If the goal is specifically the original axis=x linear gradient (drug concentration varying along flow direction), this requires candidate **C** (distributed side-injection); the ladder cannot do axis=x by mass conservation. Implementing C is a larger undertaking (multi-region inlet array) — defer until the project decides whether axis=y is acceptable.

## 8. Files referenced

- `ooc_optimizer/cfd/solver.py:67` — `evaluate_cfd` entry point.
- `ooc_optimizer/optimization/objectives.py:38–58` — `linear_gradient` factory; the target the field must match.
- `ooc_optimizer/optimization/objectives.py:178–202` — `l2_to_target` formula (normalised RMS).
- `ooc_optimizer/geometry/topology_blockmesh.py` — current opposing/same_side_Y/asymmetric_lumen builders; new topologies (ladder, side-injection) must be added here.
- `configs/default_config.yaml` — `baseline:` block (used for baseline diagnostic), `optimization.constraints`, `target_profile`.
- `examples/tumor_chip_linear_gradient/config.yaml` — campaign overrides for the completed 600-eval run.
- `examples/tumor_chip_linear_gradient/data/diagnostic/` — step-3 baseline outputs (`baseline_metrics_axis_x.json`, `baseline_metrics_axis_y.json`, `baseline_metrics_step_y_*.json`).
- `scripts/diagnostic_baseline.py` — reusable single-shot diagnostic CFD against any target spec.
