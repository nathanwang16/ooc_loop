# Interpretability

Module 3.3 is the primary scientific contribution of v2.  The BO delivers
*a* geometry that matches a target concentration profile; the
interpretability layer answers the two questions that are actually useful
to other labs:

1. **Which geometric parameters control the field?**  (global + local
   sensitivity)
2. **How precisely does each parameter need to be held during
   fabrication?**  (tolerance intervals)

All three analyses run on the **trained GP surrogate**, not directly on
CFD.  The GP is a validated surrogate over the sampled region (its
predictive accuracy is measured during BO), and evaluating it at ~1024
Saltelli points takes seconds — versus ~15 h of CFD.  The safety net is
that the top-k tightest and loosest tolerance intervals can be validated
against actual CFD re-runs (`validate_with_cfd` callback); disagreements
beyond the GP error bar indicate the sampling needs to be expanded.

## Global sensitivity — Sobol indices

Uses [`SALib`](https://salib.readthedocs.io) with a Saltelli sample.  We
report both first-order (`S1`) and total-order (`ST`) indices with
bootstrap confidence intervals.  Parameters with `S_T ≪ max(S_T)` are the
ones labs can relax.

\[
S_i \;=\; \frac{\mathrm{Var}_{x_i}[\mathrm{E}_{x_{\sim i}}(Y \mid x_i)]}{\mathrm{Var}(Y)}
\qquad
S_{Ti} \;=\; \frac{\mathrm{E}_{x_{\sim i}}[\mathrm{Var}_{x_i}(Y \mid x_{\sim i})]}{\mathrm{Var}(Y)}
\]

## Local sensitivity — GP gradients at the optimum

The BoTorch SingleTaskGP with a Matérn-5/2 kernel admits a closed-form
posterior mean gradient, which we compute via `torch.autograd` in one
backward pass.  Because the GP is trained on normalised inputs
(`x_norm ∈ [0, 1]`), `|∂μ/∂x_norm|` is already the
"change in objective per full range of parameter" — no extra scaling is
needed.

## Tolerance intervals

For each active parameter we bisect along the GP posterior mean until the
predicted objective degrades by `loss_tolerance` (default 10%) and report
the resulting `±Δx` in both normalised and physical units.

The design deliverable is a short markdown note per BO winner:

```
## Dominant parameters (global sensitivity)
- `W`, `Q_total`

## Parameters that can be held loosely
- `theta`

## Tightest fabrication tolerances
| Parameter | −Δ (phys) | +Δ (phys) |
| `W`       | 45 μm    | 55 μm     |
```

Produced automatically by
`ooc_optimizer.interpretability.pipeline.analyse_winner` as
`design_heuristics.md` inside the BO state directory.

## API

::: ooc_optimizer.interpretability.sobol.compute_sobol_indices
::: ooc_optimizer.interpretability.gp_gradients.compute_gp_gradients
::: ooc_optimizer.interpretability.tolerance.compute_tolerance_intervals
::: ooc_optimizer.interpretability.pipeline.analyse_winner
