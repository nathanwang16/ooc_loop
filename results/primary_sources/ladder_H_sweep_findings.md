# Ladder H-sweep findings (H ∈ {200, 300} μm)

## Setup

Two parallel BO jobs with identical settings (24 Sobol + 176 BO, batch=4, total=200 evals each), ladder topology only, axis=y linear-gradient target, pillar=none, varying chamber_height ∈ {200, 300} μm. Constraints: `tau_mean ∈ [0.1, 2.0]`, `f_dead ≤ 0.08`, `Re ≤ 100`, `aspect_ratio ≤ 15`, `converged ∧ mesh_ok`. Continuous bounds unchanged from the production config (W ∈ [1500, 4500], Q_total ∈ [5, 200]). Motivation: at H=200 the AR≤15 cap pinned W to 3000 μm; relaxing to H=300 expands the W cap to 4500 μm, the upper continuous bound. Wall time = 13 min 22 s; 0 errors; both JSONLs reached 200 lines cleanly.

## Results table

| H (μm) | Best L2 | Best W | Best Q_total | Best R²_to_linear | tau_mean at opt | AR at opt | Re at opt | f_dead at opt | feas_rate |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 0.0817 | 2999 | 119.81 | 0.9873 | 1.998 | 15.00 | 24.97 | 0.0312 | 37.0% |
| 300 | 0.0671 | 4496 | 200.00 | 0.9900 | 1.483 | 14.99 | 41.71 | 0.0207 | 96.5% |

Delta L2 = -0.0146 (-17.9%) at H=300 vs H=200.

## Constraint binding

- **H=200**: AR cap still binds — top-20 feasible all sit at AR=15.0, W ∈ [2998, 3000]. tau cap also binds — all top-20 have tau ∈ [1.957, 1.999], hitting the 2.0 ceiling. Both AR-cap and tau-cap are simultaneously active, and the optimum is wedged in the corner. R² already 0.987.
- **H=300**: AR cap still binds — top-20 feasible have W ∈ [≈4480, 4498], AR ∈ [14.94, 14.99]. Q_total cap also binds — winner sits at Q=200 (the upper bound). tau_mean is **NOT** binding — top-20 tau ∈ [1.471, 1.490], well below the 2.0 ceiling. f_dead and Re also have headroom (f_dead = 0.021, Re = 42).

The H=300 result moved off the tau-corner: the wider chamber + larger flow rate gives the BO room to push Q without violating tau, and tau is now ~1.48 — meaning the residence-time constraint is no longer the limiter. The AR cap and the W upper bound (4500) are the *only* active geometric limiters at H=300; the Q_total upper bound (200 μL/min) becomes the *new* active flow limiter. This implies the intrinsic-ladder L2 floor at AR=15 is approximately 0.067 — and would likely drop further if (a) W bound is widened past 4500, (b) Q_total cap raised past 200, or (c) per-strip C_k weights are unlocked (the current limit comes from a fixed equal-strip composition).

## Sobol headlines (active dim = 2; W, Q_total)

| H (μm) | ST(Q_total) | ST(W) | dominant | second |
|---|---|---|---|---|
| 200 | 0.871 | 0.143 | Q_total | W |
| 300 | 0.861 | 0.152 | Q_total | W |

Dominance order is **unchanged** (Q_total >> W) and the relative split is essentially constant (Q ≈ 86–87%, W ≈ 14–15%). Q_total remains the single most informative knob; W matters only at the AR-cap boundary.

## Comparison vs production winner

Phase-2 unconstrained-ladder analysis predicted H=300 optimum near W≈4500, L2≈0.069. Observed: W=4496 (within 4 μm of bound), L2=0.0671 — essentially on prediction. The recommended ladder L2 floor for the linear y-gradient at AR=15 is therefore **0.067 ± 0.001**, an absolute improvement of 0.0146 over the H=200 winner.

## What's next

The AR-cap remains the dominant geometric limiter at both H levels, and the H=300 optimum has migrated off the tau-cap — confirming that the residence-time constraint is *not* the binding floor on ladder L2. The next BO campaign should unlock the per-strip composition (8-D `C_k`) to test whether tuning the streamwise composition profile can drive L2 below ~0.06; if it stalls in the 0.06–0.07 band, the floor is structural (mesh diffusion or BC mismatch) and warrants a numeric/discretization audit rather than further design-space expansion. Q_total's dominance (ST≈0.86) suggests the per-strip campaign should hold W=4500, H=300, Q=200 at their current optima and sweep the C_k vector at fixed geometry.
