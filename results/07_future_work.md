# 7. Future Work — Tiered Roadmap

The campaigns reported in `04` and `05` constitute a clean stopping point for a publication or poster. This document lists the natural next steps, grouped by tier (cost / scientific yield) so a follow-up team can pick up at the appropriate level.

---

## Tier 1 — Tractable extensions (≤ 1 day each, single-line config or short patch)

These three move the project from "L2 = 0.0671 inside the constraint box" to "L2 ≈ 0.04 with a fully-converged residual budget". The minimum useful next move is **#2 then #1** (~ 1 day total), which would push L2 to ~ 0.04 and give a final tolerance/sensitivity report on a single optimal design — appropriate for the manuscript.

### 1. Per-inlet `C_k` 8-D BO at fixed (W = 4500, H = 300, Q = 200)

The H = 300 winner has W and Q both pinned at upper bounds — geometry/flow knobs are exhausted. The only remaining lever is **boundary-condition fidelity**. Letting the BO choose 8 monotonic `C_k ∈ [0, 1]` values (instead of the fixed midpoint convention) lets it pre-compensate for cross-stream diffusion that smears the imposed inlet ladder.

- **Implementation:** ~ 1 afternoon. Extend `PARAMETER_ORDER` for the ladder topology to include `C_0, ..., C_7`, mask out the existing 7 dims, add 8 active C_k dims with monotonicity constraint. Reuses the existing CFD path and BO loop.
- **Expected L2 floor:** ~ 0.04 — the analytical within-strip step-quantisation residual (0.063) plus a small numerical-diffusion residual (~ 0.01) minus the BO's pre-compensation gain (~ 0.025).
- **Why this earns its keep:** the within-strip step-quantisation is the largest *recoverable* contribution to the H = 300 residual budget (`04_optimization_results.md` §2.4). The BO pre-compensation should slightly *non-linear-ise* the inlet ladder to undo the diffusive smear, recovering most of the 0.025–0.04 gap.

### 2. Open the search box

Raise `Q_total_max` from 200 → 400 μL/min (typical syringe-pump capacity is ~ 1000) and add `H = 400` as a discrete level. This pushes the AR = 15 cap to W ≤ 6000 μm and gives BO more genuine optimisation room.

- **Implementation:** YAML-only changes.
- **Expected drop:** ~ 10 % below 0.067 just from the Q-cap relaxation; an additional ~ 5–10 % from H = 400 if the τ-cap doesn't re-bind.
- **Should be done before #1** to confirm whether L2 < 0.067 is tractable just by box-opening (i.e. without the higher-effort C_k campaign).

### 3. Mesh refinement

`ny_per_mm: 25 → 60`, with a sharper advection scheme (`bounded Gauss limitedLinear 1`). Drops the numerical-diffusion contribution to L2 by ~ 50 %.

- **Implementation:** YAML-only for the mesh; the limitedLinear scheme caused SIGFPE at high Pe in the 1-D verification (recorded in `tip.md`) and needs targeted testing.
- **Cost:** 3× per-eval wall time, so reserve for a final "publication-quality" rerun on the established winner from #1 or #2.
- **Expected drop:** ~ 0.005–0.01 below the post-#1 floor.

---

## Tier 2 — Methodological extensions (~ 1 week each)

These broaden the *applicability* of the methodology rather than tightening the L2 of the existing design.

### 4. Other target shapes

Run the full pipeline against `step axis=y` with `sharpness ∈ {0.05, 0.20}`, `bimodal axis=y` with `peak_fracs = [0.3, 0.7]`, and `linear_gradient axis=y` with shrunk range (`c_high = 0.7, c_low = 0.3`). The pipeline supports them; the AR cap at 15 may bind differently against a sharp step than against a smooth ramp.

- **Why this matters:** demonstrates the methodology generalises beyond the linear case. The admissibility table in `01_problem_and_principles.md` §1.6 predicts the ladder topology can produce all three shapes; running them validates the prediction.

### 5. Other diffusivities (large biologics)

For biologics (peptides, antibodies, ~ 100 kDa), `D` drops to `10⁻¹¹–10⁻¹² m²/s`, raising Pe by 1–2 orders. The optimal Q_total for a high-Pe biologic gradient is *lower* than 200 μL/min — the present winner does not transfer.

- **Implementation:** one-line YAML edit (`scalar_transport.D`). Re-run the H = 300 ladder BO.
- **Expected:** the Pe regime dominates the response; the new optimum will be at a *lower* Q_total and possibly a wider chamber. Will produce a different cross-topology Sobol heuristic for biologics.

### 6. Cell-line-specific reruns

If the experimental partner uses sensitive cells, re-run ladder with `tau_mean_max = 1.0 Pa`. The H = 300 winner already lies inside that tighter window; H = 200 does not.

- **Why this matters:** documents a "biology-aware" variant of the design. A reviewer will ask "does the answer change if τ ≤ 1.0?" — better to have run the campaign than to argue it.

---

## Tier 3 — Major directions (≥ 2 weeks each, separate research projects)

These are publishable as standalone follow-up papers.

### 7. Topologies B, C, D, E — implementation

Each of the four screened-but-unimplemented topologies (`christmas_tree`, `side_injection`, `permeable_wall`, `counter_flow`) requires 1–1.5 days of mesh / BC engineering plus a 200-evaluation BO run. Priority order:

- **C `side_injection`** — the **only** path to a true axis-`x` linear gradient if a follow-up project ever needs it (e.g. spheroid-PK studies that read out along the flow direction).
- **B `christmas_tree`** — the fab-realistic version of the ladder for chips that have only 2 reagent reservoirs. Worth implementing once A is published.
- **D `permeable_wall`** — requires a custom OpenFOAM Robin-type BC and possibly a multi-region solver. Significant development; defer until a project specifically needs the membrane mechanism.
- **E `counter_flow`** — not pursued. Steady-state existence is conditional on Re; high vortex-shedding risk. Recommended for a transient-CFD project, not the steady-BO pipeline.

### 8. 3-D CFD validation

The 2-D approximation collapses the floor/ceiling boundary layers into the implicit `H` parameter. Running the 3-D validation module (`ooc_optimizer/validation/cfd_3d_v2.py`) against the H = 300 winner would quantify the 2-D assumption's error in `f_dead` and `tau_mean` (the two metrics most affected by 3-D effects). Expected error: a few percent on `tau_mean`, possibly more on `f_dead` at narrow channels.

### 9. Density-based topology optimization (Borrvall–Petersson 2003)

A complete topology-optimization formulation in Brinkman-penalised Navier–Stokes that lets the *geometry itself* be optimised, not just parametric perturbations within a fixed topology class. Publishable as a standalone paper. Explicit roadmap in `primary_sources/diagnostic_findings.md` §4 and the manuscript's future-work section. Out of scope for the current cycle.

### 10. Experimental chip validation

The pipeline has not been validated against a fabricated chip. The R² = 0.990 is a CFD-vs-CFD-target metric; the chip-vs-CFD agreement is unknown until benchmark data exist. Suggested first experiment: a fluorescent-tracer inflow at the eight `C_k` values, imaged via line-scan along `y` at three downstream stations (cf. `06_translation_and_caveats.md` §1.6). This is the natural collaboration with an experimental lab.

### 11. Pillar-density sweep for the regime swap

The 1×4 pillar ablation revealed a regime swap (`Q_total` → `W`). Whether the swap accelerates further at higher pillar densities (`2×4`, `3×6`) and whether the same fixes that unlocked feasible 1×4 runs translate to the other topologies are open questions. A sweep over `pillar_config ∈ {none, 1×4, 2×4, 3×6}` × topologies would map the regime-swap as a function of pillar density and topology class. Plausibly publishable as an interpretability paper in its own right.

---

## What NOT to do

These are deliberate "don'ts" — the project has already failed at them once or has solid mechanistic reason to expect failure:

- **Do NOT re-run the original three two-inlet topologies with longer budgets, more Sobol points, or different acquisition functions.** The L2 floor on those topologies is geometric (mass conservation), not surrogate-driven; no amount of BO tuning will break it.
- **Do NOT widen the continuous bounds *without* a constraint-corner analysis first.** Bound widening tends to push the BO into corners where the constraint set was implicitly preventing degeneracies. If bounds are widened, the constraint set must be re-audited.
- **Do NOT skip Phase-1-style single-shot baseline tests for any new topology.** If the new topology cannot achieve a low L2 even at hand-picked parameters, BO will not save it. The diagnostic phase saved 600 evaluations of confirmation by running the analytical mass-conservation check first; future projects should adopt the same discipline.
- **Do NOT remove the constraint set in pursuit of lower L2.** The fully-relaxed pillar 1×4 winner (L2 = 0.0568) violates τ; reporting it without the τ caveat would be optimisation-driven over-claiming. Keep the constraint set explicit and report both constrained and unconstrained numbers when they differ.

---

## Suggested priority for a follow-up cycle

| Priority | Item | Effort | Rationale |
|---|---|---|---|
| P0 | #2 (open Q_max + H = 400) → #1 (per-inlet C_k 8-D) | ~ 1 day | Pushes L2 from 0.067 to ~ 0.04 with the existing pipeline |
| P0 | #10 (experimental chip validation) | depends on collaborator | The single missing piece for a hard publication claim |
| P1 | #6 (cell-line-specific rerun) | half-day | Pre-empts reviewer's "what if τ ≤ 1.0?" question |
| P1 | #3 (mesh refinement on the post-P0 winner) | 1 day wall + 3× per-eval cost | Final "publication-quality" rerun |
| P2 | #4 (other target shapes) | 1–2 days each | Demonstrates methodology generality |
| P2 | #5 (biologics diffusivity) | 1 day | Direct extension to a clinically distinct case |
| P3 | #7C (`side_injection` implementation) | 1.5 days | Unlocks the original axis-`x` target if needed |
| P3 | #11 (pillar-density sweep) | 1 week | Standalone interpretability paper |
| P4 | #8 (3-D validation), #9 (topology optimization) | weeks each | Major separate projects |

The P0 stack alone moves the project from "internal write-up" to "share-able formal manuscript with a publishable L2 ≤ 0.04 number."
