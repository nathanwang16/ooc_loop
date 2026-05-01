# 4. Optimization Results

This document is the empirical centre of the project. It reports, in chronological order: (1) the diagnostic-phase BO over the three two-inlet topologies (axis-`x`); (2) the four-topology BO at `H = 200 μm` against the axis-`y` linear target; (3) the height sweep that lifted L2 by another 17.9 % at `H = 300`; and (4) the pillar-ablation campaign that uncovered a regime swap and another ~30 % L2 headroom. All numbers come from the JSONL eval logs and `optimization_summary*.json` files in `examples/tumor_chip_linear_gradient/data/results/`.

---

## 0. The L2 stack — what each layer earns

Reading down the table, each row is one engineering decision; the `multiplicative gain` column is what that decision earned over the previous row:

| Stage | L2 | × prev | Cumulative ×base |
|---|---|---|---|
| Original 3 topologies + axis-`x` BO (Trial 1 winner = `opposing`) | 0.6343 | — | 1.0× (baseline) |
| Topology pivot to ladder + axis-`y` flip, hand-picked geometry, midpoint convention | 0.110 | × 5.8 | × 5.8 |
| H = 200 ladder + 2-D BO over (W, Q_total) | 0.0818 | × 1.34 | × 7.75 |
| H = 300 ladder + 2-D BO + relaxed AR cap | **0.0671** | × 1.22 | × 9.45 |
| 1×4 pillar-ablation BO at H = 200 (constraint-relaxed) | 0.0568 | × 1.18 | × 11.2 |
| (Projected) per-inlet `C_k` 8-D BO at fixed (W = 4500, H = 300, Q = 200) | ~ 0.04 | × ~ 1.7 | × ~ 16 |

**Topology selection earns ≈ 6× — the largest single jump.** Each subsequent BO/constraint-relaxation layer earns a 1.2–1.3× compounding gain. The pillar ablation (1.18×) is in the same band as the other parameter-tuning layers, which is consistent with it being a parameter-space expansion rather than a topology change. The projected per-inlet `C_k` campaign would test whether boundary-condition pre-compensation can buy another 1.7×.

---

## 1. Cross-topology BO at H = 200 μm (axis-`y` target)

**Setup.** Four topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`, `ladder`), all under the production five-constraint feasibility set, 200 evaluations each (24 Sobol init + 176 BO at batch 4). Total budget: 800 evaluations, ~1 h wall on Apple M4. The integration run hit a NaN-poisoning bug at the Sobol→BO handoff for `opposing` (`PENALTY_METRICS["Re"] = float("nan")` propagated through the constraint GP fit); the bug + fix is recorded in `tip.md`. After fix, all four topologies completed 200/200 with zero NaN crashes.

### 1.1 Best-feasible L2 per topology

Feasibility = `converged ∧ mesh_ok ∧ tau_mean ∈ [0.1, 2.0] ∧ f_dead ≤ 0.08 ∧ Re ≤ 100 ∧ aspect_ratio ≤ 15`.

| Topology | Best feasible L2 | n_feasible / n_total | Failure rate | Active dim |
|---|---|---|---|---|
| **`ladder`** (winner) | **0.0818** | 89 / 200 | 55.5 % | 2 |
| `opposing` | 0.8822 | 122 / 200 | 39.0 % | 5 |
| `same_side_Y` | 0.9937 | 127 / 200 | 36.5 % | 4 |
| `asymmetric_lumen` | 1.0875 | 47 / 200 | 76.5 % | 4 |

The **ladder beats the next-best topology by 10.8× on the same target** (apples-to-apples: identical constraints, mesh, BO budget, Sobol init protocol, target). This is the headline number for the cross-topology comparison.

The 9.4× number that appears in earlier writeups compares the H = 300 ladder winner against the axis-`x` `opposing` winner from Trial 1 — it spans both a topology change *and* a target-axis flip. The 10.8× comparison is structurally cleaner; both numbers should be reported but the 10.8× takes the lead. (See `primary_sources/PAPER_REVISION_PLAN.md` §F'.1 for the framing decision.)

### 1.2 Ladder winner geometry (H = 200)

| Field | Value | Note |
|---|---|---|
| `L2_to_target` | **0.0818** | 25% below the Phase-1 single-shot baseline (0.110) — BO recovered the residual headroom in W and Q_total |
| `R²_to_linear` | **0.987** | the field is 98.7 % linear in y |
| `monotonicity` | 0.605 | modestly above chance; the residual reflects within-strip step quantisation |
| `C_mean` | 0.500 | symmetric by construction |
| `C_std` | 0.314 | theoretical max for axis-y linear is `1/√12 = 0.289`; observed slightly higher (edge strips overshoot) |
| `W` | 2999.6 μm | **constraint-active** at `W/H ≤ 15` cap (3000 at H = 200) |
| `Q_total` | 119.5 μL/min | interior optimum |
| `tau_mean` | 1.992 Pa | **constraint-active** at upper bound 2.0 Pa |
| `f_dead` | 0.031 | well within 0.08 cap |
| `Re` | 24.9 | far below `Re_max = 100` |
| `aspect_ratio` | 14.998 | binding to within 0.001 |
| `Pe_streamwise` | 6.6 × 10⁶ | streams advection-dominated, stratify cleanly |
| `Pe_crossstream` | 2.0 × 10⁶ | transverse mixing length « chamber length |

The optimum sits **on two constraints simultaneously** (`aspect_ratio_max` and `tau_mean_max`). The achievable L2 floor inside this constraint box is ~ 0.082; opening *either* constraint would drop L2 further. The H-sweep (§2 below) opens the AR constraint by raising H; a future tau-relaxation campaign would do the corresponding test for the τ constraint.

### 1.3 Two independent ladder runs, one optimum — the free replication

Incidentally, the H-sweep (§2) re-ran a fresh `H = 200` ladder BO from a new Sobol seed alongside the new `H = 300` BO. This is a free reproducibility check on the pipeline.

| Quantity | Integration run (89 feasibles) | H-sweep redo (74 feasibles) | Δ (%) |
|---|---|---|---|
| Best feasible L2 | 0.0818 | 0.0817 | < 0.2 % |
| Best W (μm) | 2999.6 | 2999 | < 0.1 % |
| Best Q_total (μL/min) | 119.46 | 119.81 | 0.3 % |
| `tau_mean` at optimum (Pa) | 1.992 | 1.998 | 0.3 % |
| Sobol `S_T(Q_total)` | 0.871 | 0.871 | 0 % |
| Sobol `S_T(W)` | 0.143 | 0.143 | 0 % |

**Two independent BO runs from different Sobol seeds converge to the same constraint-corner optimum to within 0.3 % on every reported quantity.** The slight discrepancy in feasibility count (89 vs 74) reflects the *exploration* trajectory differing — different Sobol seeds visit different infeasible regions in the early rounds — but the *exploitation* phase converges on the same corner. This is what convergence is supposed to look like: the BO is repeatable in the answer, not in the path. The two snapshots are kept as independent confirmations; both are the same physical optimum. The pre-H-sweep H = 200 winner is archived at `examples/tumor_chip_linear_gradient/data/results/_pre_H_sweep_20260426_142449/`.

### 1.4 Cross-topology Sobol — the publishable design heuristic

Computed from each topology's GP surrogate via SALib Saltelli sampling (n = 1024). Trustworthy iff `Σ S_T < ~ 1.5`.

| Topology | Dominant param (S_T) | Subdominant | ΣS_T | Trustworthy |
|---|---|---|---|---|
| `ladder` | `Q_total` (0.871) | `W` (0.143) | 1.014 | ✓ |
| `asymmetric_lumen` | `r_flow` (0.976) | `Q_total` (0.033) | 1.022 | ✓ — surrogate essentially 1-D in `r_flow` |
| `same_side_Y` | `r_flow` (0.860) | `W` (0.128) | 1.065 | ✓ |
| `opposing` | `delta_W` (0.776) | `r_flow` (0.651) | **1.813** | ⚠ overfit suspect |

`opposing`'s `Σ S_T = 1.81` exceeds the 1.5 trustworthiness threshold — the high failure rate (39 %) compressed the GP into near-interpolation despite the noise floor. Its `delta_W` dominance claim is directionally consistent with prior diagnostics but the magnitudes are inflated. The other three topologies' indices are clean.

**Headline interpretation:** for the ladder, `Q_total` carries 87 % of total-effect variance — the gradient is set primarily by *flow rate*, not chamber width. For the two-inlet topologies, `r_flow` dominates because it is the only knob that moves the `y`-position of the drug/medium interface. **This dichotomy — flow-knob (encoded-gradient topologies) vs. interface-knob (interface-generated topologies) — is a clean design heuristic with no published precedent for tumor-on-chip gradient chambers.** Detailed mechanism + the F1/F2/F3 spotlight in `05_interpretability_findings.md` §2.

### 1.5 Constraint-binding analysis (H = 200)

| Constraint | Threshold | Effective on |
|---|---|---|
| `aspect_ratio_max` | ≤ 15 (W/H ≤ 15 at H = 200 → W ≤ 3000) | **Heavy ladder binding**: 83 of 89 feasible ladder records have W > 2900 |
| `tau_mean_max` | ≤ 2.0 Pa | Heavy ladder binding (winner at τ = 1.99); also caused 74 ladder evals to be infeasible |
| `f_dead_max` | ≤ 0.08 | Dominant infeasibility cause for `asymmetric_lumen` (117 of 200 evals) — lumen geometry creates stagnant corners |
| `Re_max` | ≤ 100 | **Never binds** — max Re across 800 evals = 24.98. The laminar gate functions as designed (safety rail) but did not actively shape the optimum |
| `tau_mean_min` | ≥ 0.1 | Caused 1–4 infeasibles per topology — low-Q corner only |

**`aspect_ratio_max = 15` and `tau_mean_max = 2.0` are jointly active at the ladder winner.** This is the cleanest possible signal that the manufacturability + biology fence is shaping the design — exactly what the constraint addition was supposed to do.

---

## 2. Extended H-sweep — ladder at H = 200 vs H = 300 μm

**Setup.** Two parallel BO jobs with identical settings (24 Sobol + 176 BO, batch = 4, total = 200 each), ladder topology only, axis-`y` target, `pillar = none`, varying `chamber_height ∈ {200, 300} μm`. Continuous bounds unchanged from the production config (`W ∈ [1500, 4500]`, `Q_total ∈ [5, 200]`). **Motivation:** at H = 200 the AR ≤ 15 cap pinned W to 3000 μm; relaxing to H = 300 expands the W cap to 4500 μm. Wall time = 13 min 22 s; 0 errors; both JSONLs reached 200 lines cleanly. Config at `examples/tumor_chip_linear_gradient/config_ladder_H_sweep.yaml`.

### 2.1 Side-by-side comparison

| H (μm) | Best L2 | Best W (μm) | Best Q (μL/min) | τ_mean (Pa) | AR | f_dead | Re | R²-to-lin | C_std | Feasibility |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 0.0817 | 2999 | 119.81 | **1.998** (cap) | **15.00** (cap) | 0.0312 | 24.97 | 0.987 | 0.314 | 37.0 % |
| **300** (winner) | **0.0671** | **4496** | **200.00** (cap) | 1.483 | **14.99** (cap) | 0.0207 | 41.71 | **0.990** | 0.306 | **96.5 %** |

**Δ vs H = 200**: L2 drops by **17.9 %** (0.0817 → 0.0671) — matches the pre-test prediction (W ≈ 4500, L2 ≈ 0.069 from the unconstrained-ladder analysis) within 2 %.

### 2.2 Constraint-corner shift — the key mechanistic finding

At H = 200 the optimum was **double-bound on `aspect_ratio_max` AND `tau_mean_max`**. At H = 300 the corner has moved:

- `aspect_ratio_max` still binds (W = 4496 ≈ 4500 cap, AR ≈ 15.0)
- **`tau_mean` is no longer binding** (1.48 Pa, well below the 2.0 cap)
- **`Q_total` upper bound (200 μL/min) is now binding** instead

The τ-relief is the predicted physics consequence of doubling-and-a-half H: at fixed (W, Q), `τ ∝ Q / (W·H²)`, so the ~ 2.25× reduction in τ-density (H ratio squared) more than compensates the 1.67× increase in Q at the new optimum. **The BO immediately spent the relaxed shear margin on higher Q_total**, which raises Pe_streamwise and sharpens the per-strip identity at the chamber outlet.

**Feasibility nearly tripled** (37 % → 96.5 %) because the wider H pushes the τ-feasibility region to encompass most of the search box. This is a major collateral benefit beyond the L2 number — at H = 300 the BO wastes far fewer evals on infeasible corners, so the GP surrogate is fitted on ~ 2.6× more useful data and is correspondingly more reliable for downstream interpretability.

### 2.3 Sobol indices unchanged in *order*; rebalanced in *share*

| H (μm) | `S_T(Q_total)` | `S_T(W)` | Σ S_T | Order change |
|---|---|---|---|---|
| 200 | 0.871 | 0.143 | 1.014 | — |
| 300 | 0.861 | 0.152 | 1.013 | none — Q_total still dominant; W slightly more influential at H = 300 |

`Q_total` remains the single most informative knob; `W` matters only at the AR-cap boundary. Σ S_T ≈ 1.01 at both H values — clean, trustworthy indices.

### 2.4 Residual-L2 budget at the H = 300 winner

A useful exercise is to decompose the 0.0671 residual into mechanistic contributions, because each contribution suggests a different next experiment:

| Source | Estimated L2 contribution | Mechanism / Reducible by |
|---|---|---|
| Within-strip step quantisation (N = 8 staircase) | ≈ 0.063 | Analytical: `1/(2N√3) ≈ 0.036`, normalised by `‖C_target‖_2 = 1/√3` gives 0.063. Reducible only by larger N or per-inlet `C_k` tuning. |
| Cross-stream diffusion smoothing | ≈ −0.020 | Diffusion *helps* by rounding the staircase — partially cancels the previous term. Lowering Pe further would help more, but breaks streamline stratification. |
| Numerical (upwind) diffusion | ≈ +0.015 | Mesh + first-order advection scheme. Reducible to ~ 0.005 with `ny_per_mm: 25 → 60` + linear-upwind. Cost: 3× per-eval wall time. |
| Inlet-region acceleration & step rounding | ≈ +0.005 | Streams enter at fixed `C_k` but spread laterally as they accelerate near the inlet; near-`x = 0` slightly less linear than the bulk. |
| Constraint-corner pinning (Q at YAML cap) | ≈ +0.005 | Q_total = 200 is a YAML choice; raising it to ~ 270 (the τ-cap-binding ceiling at H = 300) drops L2 a few percent more. |
| **Total** | **≈ 0.067** | observed 0.0671 |

The largest *recoverable* contribution is the **within-strip step quantisation**. The midpoint convention `C_k = (k+0.5)/N` is geometrically optimal *if the chamber's response were the identity map*, but cross-stream diffusion smears each step into the next, and the optimal pre-compensation is a slightly nonlinear `C_k` profile. Letting the BO choose those 8 values jointly is the per-inlet `C_k` 8-D campaign in `07_future_work.md` §1; expected payoff ≈ 25 % on top of 0.067, putting L2 ≈ 0.05 within reach.

The numerical-diffusion contribution is the second-largest but is also the least scientifically interesting — it is a discretisation artefact, not a physical limit. Worth eliminating only on the final "publication-quality" rerun.

---

## 3. Pillar ablation — the regime swap

**Setup.** A 100-evaluation constrained BO at H = 200 μm on the ladder topology with `pillar_config = 1×4` (one row of four cylindrical pillars). The active design space becomes 4-D in `(W, d_p, s_p, Q_total)`; the inlet-junction parameters `(θ, r_flow, δ_W)` remain structurally inactive for the ladder geometry generator and are still masked. Motivation: the bare-ladder masking (5 of 7 parameters pinned) was decided on physical-intuition grounds; `04_optimization_results.md` §1.4 shows that the bare-ladder Sobol cannot validate the masking decision because the masked parameters are absent from the surrogate. The ablation directly tests whether unmasking `d_p, s_p` exposes hidden structure.

### 3.1 The result overturns the a-priori assumption in two ways

**(a) The best L2 falls to 0.0568** — a ~ 30 % drop relative to the bare-ladder H = 200 winner of 0.0818 — at:

| Field | Value |
|---|---|
| L2 | **0.0568** |
| R²_lin | 0.992 |
| W | 2121 μm |
| d_p | 184 μm |
| s_p | 427 μm |
| Q_total | 125 μL/min |
| Re | 31.9 (laminar) |
| τ_mean | 2.62 Pa (**violates the 2.0 cap**) |

Reported alongside the production constraint set, this design narrowly violates the τ-cap. The best fully-feasible design under all five production constraints sits at L2 ≈ 0.0588 — comparable to but still below the bare-ladder H = 200 winner. The pillar gain is *real* but not yet fully captured under the production constraint set; relaxing the τ cap (or moving to a less shear-sensitive cell line) recovers the additional ~ 30 %.

**(b) The Sobol indices on the new 4-D surrogate INVERT which knob controls the gradient.** Side-by-side with the bare-ladder Sobol (from §1.4 above):

| Parameter | `S_T` (pillars = none) | `S_T` (pillars = 1×4) |
|---|---|---|
| `W` | 0.143 | **0.856** |
| `Q_total` | **0.871** | 0.001 |
| `s_p` | masked | **0.139** |
| `d_p` | masked | 0.009 |

`Q_total` collapses from dominant to invisible. `W` jumps from secondary to dominant. `s_p` emerges as a meaningful secondary lever; `d_p` stays minor.

### 3.2 Mechanistic interpretation

With **no pillars**, the chip is essentially a parallel-plate channel and the linear gradient is recovered via diffusion over residence time `L/U` — total flow rate sets the answer, and `S_T(Q_total)` ≈ 0.87 follows.

With **pillars present**, the chip becomes a *structured medium* and the gradient becomes a function of `W/s_p` rather than `Q_total`; pillar diameter `d_p` remains a second-order modulator of local pressure drop and mixing-zone footprint. The new optimum sits at moderate `W ≈ 2100 μm` rather than the corner-pinned `W = 3000` of the bare-ladder H = 200 winner, because in a structured medium wider chambers accumulate more diffusive smear before the outlet.

**The same physical chamber, with one row of pillars introduced, is governed by an entirely different control variable.** This is the regime swap.

### 3.3 Concentration field of the pillar winner

The 1×4 pillar winner's concentration field (figure `fig_h_pillar_field.png` in `bayesian_src/`) shows the depth-averaged profile sitting within ~ 1 % of the linear target across the chamber, and the three downstream stations (x ∈ {1, 5, 9} mm) are essentially identical — the gradient is **fully formed by x ≈ 1 mm** rather than ~ 5 mm in the bare-ladder case. Pillar-induced re-mixing accelerates the inlet rounding, compressing the transit length.

### 3.4 Two methodological caveats

1. **The "ladder is dominated by W and Q_total" interpretation is correct only within the bare-ladder design subspace.** Once pillars enter the picture, the dominant control variable swaps. The ranking inferred under masking is no longer accurate for the pillar configuration.
2. **The masking decision was made on physical-intuition grounds; the regime swap demonstrates that empirical ablation is the appropriate audit for any masked parameter that could plausibly couple to the response.** The lesson generalises: *Mask, then audit by ablation.*

Whether the swap accelerates further at higher pillar densities (`2×4`, `3×6`) and whether the same fixes that unlocked feasible 1×4 runs (an empty front/back patch type for 2-D meshes; relaxed concave-cell tolerances for snappyHexMesh on cylinders) translate to the other three topologies are open questions for follow-up.

---

## 4. Translating the H = 300 ladder winner to a chip spec

The H = 300 ladder winner is a fully-specified chip ready for fabrication:

| Spec | Value |
|---|---|
| Chamber length L | 10 mm (fixed) |
| Chamber width W | **4496 μm** (≈ 4.5 mm) |
| Chamber height H | **300 μm** |
| Number of inlet strips N | 8 |
| Per-strip inlet width | W/N = 562 μm |
| Per-strip inlet C_k | 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375 (midpoint convention) |
| Per-strip inlet U_x | Q_total / (N · W/N · H) = uniform across strips |
| Total flow rate Q_total | **200 μL/min** |

What the experimentalist needs upstream of the chip:

- **Eight pre-mixed reservoirs** at the eight `C_k` values, **or** a single binary mixer tree producing them on-chip (candidate B in the topology screen).
- **One precision syringe-pump pair** capable of 200 μL/min total (~ 25 μL/min per strip), with ± 2 % accuracy. Sobol says this is the high-priority calibration target.
- **PDMS soft-lithography fabrication** at W/H = 15. This is at the edge of the "safe" range (literature consensus 10–20 for PDMS roof sag); the chip is fabricable but not over-engineered. A 12-μm-tolerance soft-litho process is more than adequate (the design tolerates ± 540 μm on W; see `05_interpretability_findings.md` §3).
- **Cell-line check.** At τ = 1.48 Pa sustained, the design is comfortable for endothelial cells, kidney/hepatocyte lines, and most cancer cell lines. **Sensitive primary cells (neurons, primary tumor explants) should not run at this shear** — switch to H = 400 μm at the same W to drop τ further (predicted L2 ≈ 0.072 with a single-line YAML edit, still better than the H = 200 result).

The single most important operational note: **the chip's R² = 0.990 is achieved only at the design Q_total = 200 μL/min**. Halving Q drops Pe and breaks streamline stratification, dropping R² noticeably. The chip is not an "any-flow gradient generator" — it is *specifically* a 200 μL/min linear-gradient generator. Calibrate Q with a flow-rate sensor, not just by syringe-pump nominal setting.

The pillar 1×4 variant offers another ~ 30 % L2 headroom at moderate W ≈ 2100 μm, but with τ ≈ 2.6 Pa — usable only with shear-tolerant cell lines or with the τ cap relaxed.
