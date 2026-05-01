# 3. Topology Screening — Eight Topologies, Two Trials

The project considered eight topologies in total across two design trials. Three were carried over from the original axis-`x` campaign; five were proposed after the diagnostic phase identified the mass-conservation limit. Of the five new candidates, only the stacked ladder was implemented end-to-end; the other four were screened in or out by first-principles analysis without ever running CFD on them. This document records the screening logic so a follow-up project can reuse the rationale or pick up an unimplemented candidate.

---

## 3.1 The eight topologies, mapped to trials and analyses

| Topology | Trial | Target axis | Active dim | Analysis applied | Status |
|---|---|---|---|---|---|
| Opposing | 1, 2 | x, y | 5 | full BO at H = 200 (axis-x and axis-y) | implemented; archived axis-x winner; axis-y baseline in `04` |
| Same-side Y | 1, 2 | x, y | 4 | full BO at H = 200 (axis-x and axis-y) | implemented; baseline in `04` |
| Asymmetric lumen | 1, 2 | x, y | 4 | full BO at H = 200 (axis-x and axis-y) | implemented; baseline in `04` |
| **Ladder** | 2 | y | 2 (bare) / 4 (with pillars) | full BO at H = 200 + H = 300; pillar ablation at H = 200 | implemented; **winner** |
| Christmas-tree mixer | 2 | y | (would be ~6) | first-principles screening only | future work — admissible |
| Distributed side injection | 2 | x | (would be ~5–10) | first-principles screening only | future work — only admissible axis-x route |
| Permeable wall | 2 | x | (would be ~3–5) | first-principles screening only | future work — needs membrane model |
| Counter-flow | 2 | x | (would be ~3) | first-principles screening only | not pursued — unsteady-state risk |

"BO" denotes a full BoTorch CEI campaign with ≥ 100 evaluations. "Screened only" denotes first-principles admissibility analysis without CFD enumeration.

---

## 3.2 The three two-inlet baselines (Trial 1, carried over to Trial 2)

These three were the original design space when the project began. All three share the same boundary structure — two coflow inlets at `x = 0`, single outlet at `x = L`, no through-wall scalar source — so they are subject to the same mass-conservation limit (`01_problem_and_principles.md` §1.5). They differ only in the *spatial arrangement* of the inlets and chamber walls.

### 3.2.1 Opposing

- **Geometry:** Two short-side inlets at `x = 0`, separated by a tongue partition that splits the chamber width into two upstream feed channels. Tongue offset `δ_W ∈ [0.12, 0.48]` (normalised) parameterises where the partition starts.
- **Active dims:** `(W, Q_total, r_flow, θ, δ_W)` = 5.
- **Achievable target:** transverse step in `y` (the interface between the two streams stratifies along `y`); cannot achieve any linear gradient in `x`.
- **Trial-1 result (axis-x):** L2 = 0.6343 — the diagnostic-phase winner across the three two-inlet topologies, sitting 8 % above the uniform-field floor.
- **Trial-2 result (axis-y):** L2 = 0.8822 (worse than ladder by 10.8×) — the same mass-conservation limit that capped axis-x bounds it on axis-y too within the laminar regime sampled.

### 3.2.2 Same-side Y

- **Geometry:** A Y-junction splits a single inlet into two half-height inlets that run parallel before entering the chamber. Both at `x = 0`, vertically stacked.
- **Active dims:** `(W, Q_total, r_flow, θ)` = 4.
- **Trial-1 result (axis-x):** L2 = 0.8296. `r_flow` wedged at upper bound (0.97).
- **Trial-2 result (axis-y):** L2 = 0.9937. Behaviour consistent with diagnostic — `r_flow` dominates Sobol, but the achievable optimum is poor.

### 3.2.3 Asymmetric lumen

- **Geometry:** Ayuso-2020-style lumen on a side wall — one inlet at `x = 0` is the bulk flow, a second smaller inlet enters via a side-wall lumen further downstream.
- **Active dims:** `(W, Q_total, r_flow, θ)` = 4.
- **Trial-1 result (axis-x):** L2 = 0.7351; all 200 evaluations feasible.
- **Trial-2 result (axis-y):** L2 = 1.0875; **76.5 % infeasible** because the lumen geometry creates stagnant corners that the BO cannot reduce below `f_dead_max = 0.08`. This is a topology-intrinsic limitation, not a BO-tuning issue.

**Cross-topology Sobol on the trio is the source of findings F1, F2, F3** (`05_interpretability_findings.md` §2):

- F1: `r_flow` is the universal driver (`S_T = 0.86–0.98`; `opposing` 0.65 with caveat).
- F2: `θ` is consistently inactive (`S_T = 1e-4 to 1e-3` everywhere) — future studies can fix `θ` at midpoint and save a dimension.
- F3: `W` and `Q_total` trade off as the secondary lever, with the ranking flipping by topology.

---

## 3.3 Trial 2 — the five replacement candidates

Five candidates were proposed and analysed from first principles after the diagnostic phase showed the trio could not produce a linear-x gradient. The screening criteria were:

1. **Mass-conservation compatibility** with the desired target.
2. **Achievable Pe regime** (advection-dominated to keep streamline stratification, but with enough cross-stream diffusion length to smooth out imposed steps).
3. **Manufacturability** under PDMS soft-lithography.
4. **BC-engineering complexity** in OpenFOAM (multi-region, custom BCs, time stepping).
5. **Steady-state existence** (not all candidates have a steady-state regime that the project's BO framework can score).

Each candidate is summarised by its mechanism, why it works (or doesn't), implementation cost, and verdict.

### 3.3.1 A — Stacked ladder (the implemented winner)

- **Achievable target:** linear gradient *transverse to flow* (axis = `y`), step in `y`, bimodal in `y`.
- **Mechanism:** N stacked inlet strips at `x = 0`, each at prescribed concentration `C_k = (k+0.5)/N` (midpoint convention). The N parallel laminar streams flow downstream side-by-side; transverse diffusion smooths the discrete steps into a quasi-linear `y`-profile after a mixing length `L_mix ≈ W² u / D`.
- **Why it works:** at `x = 0` the gradient is *imposed* (a ladder that is already linear in `y` on average); the chamber's job is just to smooth the steps without destroying the trend. Mass conservation is satisfied because the gradient is transverse to flow, not along it.
- **Implementation:** extends `ooc_optimizer/geometry/topology_blockmesh.py` with `_bm_ladder` — N inlet patches; the BC writer (`cfd/solver.py:_setup_case`) assigns `fixedValue C = C_k` per patch and uniform per-strip `U_x`. N = 8 is the canonical Whitesides choice; `N = {4, 8, 16}` were all tested in the diagnostic phase.
- **Risks:** streams must remain laminar (`Re < 100`, satisfied); inlet fabrication cost rises with N; N = 8 is a sweet spot.
- **Verdict:** ✓ Implemented. Winner across all four trial-2 BO campaigns.

### 3.3.2 B — Christmas-tree mixer

- **Achievable target:** linear `y`-gradient, same as A, but with on-chip preparation of the N intermediate concentrations from only two physical reagent reservoirs (drug + medium).
- **Mechanism:** a binary tree of serpentine mixing channels splits the two reagents repeatedly, producing N = 2^k parallel outputs at concentrations spaced by `1/(N−1)`. These then feed the wide chamber as in A.
- **Why it works:** same as A, but solves the practical fabrication problem of needing N reagent streams when only two are available off-chip.
- **Implementation:** significant geometry work — the mixer tree itself must be meshed and solved. The serpentine channels require full mixing before each junction (channel length scales as `W² u / D ≈ several mm` per branch); total tree footprint can dwarf the chamber. Per-evaluation CFD cost roughly doubles.
- **Risks:** mesh complexity; numerical diffusion in serpentine channels can degrade the prepared gradient before it reaches the chamber.
- **Verdict:** ✓ Admissible. Future work after A was demonstrated.

### 3.3.3 C — Distributed side injection (the only axis-x route)

- **Achievable target:** linear gradient *along flow* (axis = `x`) — the *idealised* original target.
- **Mechanism:** an array of K small inlets along the chamber side wall (`y = 0`) injecting `C = 1`, plus the original `x = 0` inlet at `C = 0` feeding the bulk medium. Each side-inlet adds drug mass locally; with prescribed flow rates `Q_k(x_k)` increasing along `x`, the cumulative drug fraction at position `x` is

  $$\frac{\sum_{i:\, x_i < x} Q_i}{Q_{\text{medium}} + \sum_{i:\, x_i < x} Q_i}.$$

  With proper `Q_k` spacing this *can* be linear in `x`.
- **Why it works:** the side inlets are the distributed scalar source the conservation argument required. Mass conservation is no longer violated because mass genuinely enters through the side wall.
- **Implementation:** extends topology builder with K side-inlet patches (parametric `K ∈ {4, 8, 16}`). Total inlet flow varies along `x` → outlet flow ≈ `Σ Q_k`, much larger than any single inlet. Solver handles fine; mesh complexity moderate.
- **Risks:** each side jet creates a local recirculation; total `f_dead` will rise above the current 0.08 threshold; constraint must be relaxed. K is a discrete BO variable adding combinatorial complexity. Fabrication cost in a real device is high (many parallel pumps).
- **Verdict:** ✓ Admissible — the **only** candidate that produces the actual axis-x linear gradient. If a follow-up project requires the original target axis (e.g. spheroid-PK studies that read out along the flow direction), this is the path.

### 3.3.4 D — Permeable wall with reservoir

- **Achievable target:** linear `x`-gradient, smoother than C.
- **Mechanism:** replace one chamber wall (e.g. the floor at `z = 0`) with a permeable membrane separating the chamber from a drug reservoir at `C = 1`. The flux through the membrane at position `x` is `D_m · (C_reservoir − C(x, y = 0, z = 0)) / δ_m`; this is `x`-dependent because `C` in the chamber is `x`-dependent. With a *graded* membrane permeability `K(x) = K_0 · x/L`, the flux ramps along `x` and the chamber mean concentration grows roughly linearly with `x`.
- **Why it works:** distributed source via wall flux. The grading is in fabrication, not in flow control — only one reservoir is needed.
- **Implementation:** requires a custom OpenFOAM BC (Robin-type) and possibly a multi-region solver (`chtMultiRegionFoam` or similar) if the membrane diffusion is co-resolved. Significant development outside the current scope.
- **Risks:** high-effort prototype before knowing whether grade-K(x) fabrication is feasible.
- **Verdict:** ✓ Admissible but expensive. Defer.

### 3.3.5 E — Counter-flow inlets

- **Achievable target:** weak linear `x`-gradient, more like erf-shape.
- **Mechanism:** drug enters at `x = 0` at low flow rate, medium enters at `x = L` at high flow rate; they meet somewhere inside the chamber; net flow exits through side-wall outlets in the middle. Drug propagates upstream against medium by diffusion, creating a *frontal interface* whose position depends on flow ratio. The interface zone has `x`-dependent `C`.
- **Why it works:** counter-flow creates an x-dependent interface that is structurally distinct from the two-inlet coflow.
- **Risks:** fundamentally unsteady regimes are easy to fall into (vortex shedding at the impingement). Steady-state existence is not guaranteed for all parameter combinations. Likely poor fit for the BO penalty framework, which assumes well-defined steady metrics.
- **Verdict:** ⚠ **Unsteady-state risk; not pursued.** Steady-state existence conditional on Re; would require a separate transient-CFD pipeline outside the current scope.

---

## 3.4 The screening table — admissibility, ranked

A consolidated screening view, ranked by recommended order of investment:

| Rank | Topology | Target | Cost | Likely L2 floor | Notes |
|---|---|---|---|---|---|
| **1** | A — Stacked ladder, N = 8 | y | low | ≪ 0.10 (achieved 0.067 at H = 300; 0.057 with pillars) | cheapest demonstration; **implemented** |
| 2 | C — Distributed side injection | **x** | medium | unknown but tractable | only path that preserves the original axis-x target |
| 3 | B — Christmas-tree mixer | y | high | similar to A | only worth it after A succeeds, for practical fab |
| 4 | D — Permeable wall | x | very high | unknown | OpenFOAM development required |
| 5 | E — Counter-flow | x | low to set up | poor | unsteady risk; not recommended |

The full first-principles analysis for each of A–E was carried out before any CFD on the candidates, demonstrating that **the screening table is achievable at zero CFD cost.** That is the methodological point: a follow-up project starting from a different prescribed target would use the admissibility table from `01_problem_and_principles.md` §1.6 plus the screening criteria above to triage candidates *before* committing CFD time.
