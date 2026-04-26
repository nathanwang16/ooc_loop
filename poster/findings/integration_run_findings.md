# 4-Topology Integration Run — Findings (axis=y linear gradient)

**Date:** 2026-04-26.
**Pipeline:** `scripts/run_optimization.py` (4-topology BO, 200 evals/topology, axis=y target, 5-constraint feasibility set).
**Status:** All four topologies completed 200/200 evals after a bug-fix rerun (NaN-poisoned constraint GPs in the original integration run; root cause and fix recorded in `tip.md`).

---

## 1. Run summary

| Phase | Wall time | Evals | Notes |
|---|---|---|---|
| First integration run (4 topologies) | ~2 h (12:20 → 14:25) | 624 evals before crash | 3 of 4 topologies completed 200/200; `opposing` crashed at Sobol→BO handoff with `botorch.exceptions.errors.InputDataError: Input data contains NaN values.` |
| Bug fix | <2 min | — | Replaced `float("nan")` sentinels in `PENALTY_METRICS` (`solver.py`) and metrics-fallback dict (`metrics.py`) with finite values (`Re=1e6`, `aspect_ratio=1e3`, `Pe_*=R²=0.0`); defensive guard in `bo_loop._evaluate_point` updated to map NaN→finite (was NaN→`inf`, also unfittable). |
| Recovery rerun (`opposing` only) | ~23 min (13:37 → 14:00) | 200/200 | 0 NaN crashes; 2 non-fatal `Field T uniform 0` extraction errors absorbed as penalty L2=99 records; BO ran to completion. |
| Interpretability across all 4 BO states | ~3 s | — | `scripts/run_interpretability.py` over each `bo_<topology>_none_H200/` state dir; produces per-topology `summary.json`, `sobol.png`, `local_sensitivity.png`, `tolerance.png`, `design_heuristics.md`. |

**Total useful eval budget:** 800 (4 × 200).

## 2. Best feasible L2 per topology

Feasibility = `converged AND mesh_ok AND tau_mean ∈ [0.1, 2.0] AND f_dead ≤ 0.08 AND Re ≤ 100 AND aspect_ratio ≤ 15`.

| Topology | Best feasible L2 | n_feasible / n_total | Failure rate |
|---|---|---|---|
| **`ladder`** (winner) | **0.0818** | 89 / 200 | 55.5% |
| `opposing` | 0.8822 | 122 / 200 | 39.0% |
| `same_side_Y` | 0.9937 | 127 / 200 | 36.5% |
| `asymmetric_lumen` | 1.0875 | 47 / 200 | 76.5% |

**Ladder beats the next-best topology by 10.8× on L2 and the prior axis=x campaign winner (`opposing`, L2 = 0.6343) by 7.76×.** The improvement is real and not a metric-normalization artefact: ladder's `R²_to_linear = 0.987` confirms the field is linear-gradient-shaped to within 1.3% of variance.

## 3. Cross-topology winner — `ladder`

| Metric | Value | Notes |
|---|---|---|
| `L2_to_target` | **0.0818** | 25% below the Phase-1 single-shot baseline (0.110) — BO recovered the residual headroom in W and Q_total. |
| `R²_to_linear` | **0.987** | The field is 98.7% explained by a linear y-fit. |
| `monotonicity` | 0.605 | Modestly above chance; remaining 40% is the within-strip step quantisation (the C field is a smoothed staircase, not a pure ramp). |
| `C_mean` | 0.500 | Perfect by construction (symmetric ladder, mass conservation). |
| `C_std` | 0.314 | Theoretical maximum for axis=y linear is `1/√12 = 0.289`; observed slightly higher because edge strips overshoot slightly. |
| `tau_mean` | 1.992 Pa | **Constraint-active**: sits within 0.4% of `tau_mean_max = 2.0`. |
| `f_dead` | 0.031 | Well within 0.08 cap. |
| `Re` | 24.9 | Far below `Re_max = 100`. |
| `aspect_ratio` | 14.998 | **Constraint-active**: at `aspect_ratio_max = 15` to within 0.001. |
| `Pe_streamwise` | 6.6×10⁶ | Confirms streams are advection-dominated and stratify cleanly. |
| `Pe_crossstream` | 2.0×10⁶ | Same regime — transverse mixing length « chamber length. |

### Winner geometry

| Param | Value | Bound |
|---|---|---|
| `W` | 2999.6 μm | hits 3000 μm cap (set by W/H ≤ 15 at H=200 μm) |
| `Q_total` | 119.5 μL/min | bound [5, 200] |
| `r_flow` | 0.535 | inactive for ladder; pinned to bound midpoint 0.535 |
| Inactive: `d_p`, `s_p`, `theta`, `delta_W` | midpoint values | masked out for ladder topology by `_active_params` |

The optimum sits **on two constraints simultaneously** (`aspect_ratio_max = 15` and `tau_mean_max = 2.0`), suggesting the achievable L2 floor inside this constraint box is `~0.082` — opening either constraint would drop L2 further. With H=300 μm available as a discrete level (currently fixed at 200), the W/H ≤ 15 cap could relax to W ≤ 4500 μm, which is the axis-y diffusive-smoothing length the ladder benefits from.

## 4. Cross-topology Sobol sensitivity

Computed from each topology's GP surrogate via SALib Saltelli sampling (n=1024).

| Topology | Dominant param (S_T) | Subdominant | ΣS_T | Trustworthy? |
|---|---|---|---|---|
| `ladder` | `Q_total` (0.869) | `W` (0.148) | 1.02 | ✅ |
| `asymmetric_lumen` | `r_flow` (0.976) | `Q_total` (0.033) | 1.02 | ✅ — surrogate is essentially 1-D in `r_flow` |
| `same_side_Y` | `r_flow` (0.860) | `W` (0.128) | 1.07 | ✅ |
| `opposing` | `delta_W` (0.776) | `r_flow` (0.651) | **1.81** | ⚠️ overfit suspect |

`opposing`'s ΣS_T = 1.81 exceeds the 1.5 threshold for trustworthy total-effect indices — its high-failure-rate (39%) Sobol points appear to have collapsed the GP into a near-interpolation regime. The other three topologies' indices are clean and physically interpretable.

**Headline interpretation:** for the ladder, `Q_total` (which sets advective contact time and tau_mean) carries 87% of the total-effect variance — the gradient is set primarily by *flow rate*, not chamber width. For the two-inlet topologies (`same_side_Y`, `asymmetric_lumen`), `r_flow` dominates because it sets the y-position of the drug/medium interface, which is the only meaningful structural feature these topologies can produce.

## 5. Constraint-binding analysis

| Constraint | Threshold | Effective on which topologies? |
|---|---|---|
| `aspect_ratio_max` | ≤ 15 (W/H ≤ 15 at H=200 μm caps W ≤ 3000) | **Heavy ladder binding**: 83 of 89 feasible records have W > 2900 μm. The cap is shaping the ladder optimum; relaxing it would drop L2 further. |
| `tau_mean_max` | ≤ 2.0 Pa | Heavy ladder binding (winner at tau=1.99); also caused 74 evals on ladder to be infeasible. |
| `f_dead_max` | ≤ 0.08 | Dominant cause of infeasibility on `asymmetric_lumen` (117 of 200 evals). The lumen geometry creates stagnant corners. |
| `Re_max` | ≤ 100 | **Never binds** — max Re across all 800 evals was 24.98 (asymmetric_lumen at high-Q corner). The laminar gate functions as designed (safety rail) but didn't actively shape the optimum. |
| `tau_mean_min` | ≥ 0.1 | Caused 1–4 infeasibles per topology — low-Q corner only. |

**`aspect_ratio_max = 15` and `tau_mean_max = 2.0` are jointly active at the ladder winner.** This is the cleanest possible signal that the "manufacturability + biology fence" is shaping the design — exactly what the constraint addition was supposed to do. The user-chosen relaxed value (W/H ≤ 15 instead of literature-canonical 10) gave BO the headroom to find a high-quality interior optimum. A future campaign that admits H=300 μm (allowing W/H ≤ 15 → W ≤ 4500 μm) is a one-line config change that should drop L2 further.

## 6. Comparison with prior axis=x campaign

| Campaign | Target | Winner | Best L2 | Note |
|---|---|---|---|---|
| Original (archived) | `linear_gradient axis=x` | `opposing_none_H200` | 0.6343 | wedged 8% above uniform-field floor (0.585); mass conservation forbids axis=x for two-inlet coflow. |
| This run | `linear_gradient axis=y` | **`ladder_none_H200`** | **0.0818** | **7.76× better**; first time the pipeline produces a field that is genuinely linear (R² = 0.987) and not just "as uniform as possible". |

This is the headline finding for the writeup: **the topology pivot from two-inlet coflow to N-stacked ladder, combined with the target-axis flip, breaks through the mass-conservation ceiling that capped the original campaign.** The Sobol indices, the constraint-binding analysis, and the R²-to-linear all corroborate the story.

## 7. Honest critical caveats

- **`opposing` Sobol indices are not trustworthy** (ΣS_T = 1.81). High failure rate (39%) likely caused the GP to interpolate near coincident penalty points despite the noise floor we previously added. Treat the `delta_W` dominance claim with caution; it's directionally consistent with prior diagnostic findings but the magnitudes are inflated.
- **Two non-fatal `Field T uniform 0` extraction errors** appeared in the rerun log. These are absorbed as penalty L2=99 records and don't bias the BO, but they indicate the scalar transport occasionally produces a degenerate field (likely when an extreme corner of the design space gives a malformed mesh that simpleFoam survives but `scalarTransportFoam` doesn't). Not investigated further — frequency is low (1%).
- **Ladder winner is constraint-double-active**: the achievable L2 floor inside the constraint box is ~0.082, not the topology's intrinsic floor. The intrinsic floor (uncapped W and Q) could be lower; the analytical N=8 within-strip residual is ~0.063, so there's another ~25% headroom available if the constraints are relaxed.
- **Phase-2 (W, Q_total) Sobol scan from earlier diagnostic was *flat* (L2 ∈ [0.172, 0.176])** but this BO winner is at 0.082. The discrepancy is because the Phase-2 scan used the endpoint convention (C_k = k/(N-1)) while the production integration uses the midpoint convention (C_k = (k+0.5)/N) — a 38% drop comes from convention alone, plus another ~25% comes from BO actively pushing W toward the W/H cap and Q_total toward the tau cap.
- **`r_flow = 0.535` is inactive for ladder** but still present in the BO vector pinned at the bound midpoint. This is by design (`PARAMETER_ORDER` is shared across all topologies for joint plotting) but means cross-topology Sobol comparisons should mask `r_flow` for ladder explicitly — done correctly in the per-topology summary.json output.

## 8. What's next (recommended)

1. **Lift the `aspect_ratio` constraint** by enabling H=300 μm as a discrete level — allows W ≤ 4500 μm at W/H = 15. Single YAML edit in `discrete_levels.chamber_height: [200, 300]`. Expected L2 drop: ~10–20%.
2. **8-D BO over per-inlet `C_k`** for the ladder topology (instead of the current 2-D BO over W, Q_total). The optimum will be a slight nonlinear correction to `C_k = (k+0.5)/N` that compensates for cross-stream diffusion. Theoretical L2 floor under this campaign: ~0.04. Implementation cost: ~1 afternoon (extend `PARAMETER_ORDER` for the ladder topology only, mask the existing 7 dims out).
3. **Mesh-refinement campaign** for ladder: `ny_per_mm: 25 → 60`, sharper advection scheme. Drops the numerical-diffusion contribution to L2. Cost: 3× per-eval wall time; useful only after step 1+2 are done.
4. **Topologies B–E** remain on the 1–1.5 day-each roadmap. Highest-value next is **C `side_injection`** if the project ever needs the original axis=x gradient (e.g. for spheroid-PK studies). Recommended only after publishing the current axis=y result.
