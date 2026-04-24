# Inverse design

The v2 BO loop minimises

\[
\mathcal{L}(\mathbf{x}) \;=\; \bigl\| C_\text{sim}(\mathbf{x}) - C_\text{target} \bigr\|_2 \,/\, \bigl\| C_\text{target} \bigr\|_2
\]

subject to a small set of hard constraints that ensure the flow is
physically reasonable:

\[
\tau_\text{mean} \in [0.1, 2.0]~\text{Pa}, \qquad
f_\text{dead} \le 0.05,
\]

where `f_dead` is the fraction of the chamber floor with velocity below
10% of the floor-averaged mean.  The WSS bounds are relaxed relative to
v1 because drug transport is far less WSS-sensitive than endothelial
biology (see Development Guide v2 §6, Module 2.2).

## Target profile classes

The shipping profile types (`ooc_optimizer.optimization.objectives`) are:

- `linear_gradient(axis, c_high, c_low)` — monotonic ramp along `x` or `y`.
- `bimodal(peak_axis, peak_fracs, width_frac, c_peak, c_base)` — sum of two
  Gaussian bumps; models "rim + rim" exposure patterns.
- `step(step_axis, step_frac, sharpness_frac, c_high, c_low)` — sigmoid
  step at `step_frac · L`.
- `custom(fn, name)` — wrap any `fn(x, y, *, L, W) -> ndarray`.

YAML spec (see `configs/default_config.yaml`):

```yaml
target_profile:
  kind: linear_gradient
  axis: x
  c_high: 1.0
  c_low: 0.0

extra_target_profiles:
  - kind: bimodal
    peak_axis: x
    peak_fracs: [0.25, 0.75]
    width_frac: 0.1
  - kind: step
    step_axis: x
    step_frac: 0.5
    sharpness_frac: 0.05
```

## BO mechanics (carried forward from v1)

- Matérn-5/2 kernel SingleTaskGP with learned length scales and output
  scale.
- Sobol initialisation (`n_sobol_init = 20` in v2).
- `ConstrainedExpectedImprovement` acquisition (one constraint GP per
  constraint).
- Per-discrete-config independent BO runs — embarrassingly parallel across
  `(pillar_config × chamber_height × inlet_topology)`.
- `run_multi_target_workflow` sweeps all 24 configurations for the primary
  target profile, then runs the *winning topology only* for the two
  secondary profiles.  Total budget ≈ 1,500 forward solves; about 12 h on
  eight cores.

## Per-topology parameter masks

The BO input vector is 7-dimensional for all topologies
(`PARAMETER_ORDER` in `ooc_optimizer.optimization.bo_loop`).  Inactive
dimensions are pinned to 0.5 so the GP never sees informationless inputs:

- `pillar_config = "none"` → `d_p`, `s_p` masked out.
- `topology ≠ "opposing"` → `delta_W` masked out.

The active mask is serialised with the BO state so that interpretability
(Module 3.3) analyses only the dimensions that actually affected the
surrogate.

## API

::: ooc_optimizer.optimization.bo_loop.BORunner
::: ooc_optimizer.optimization.orchestrator.run_multi_target_workflow
::: ooc_optimizer.optimization.objectives
