# Scalar transport

After `simpleFoam` converges on the momentum equation we freeze the
velocity field and solve the passive-scalar transport equation

\[
\nabla \!\cdot\! (\mathbf{u} \, C) - \nabla \!\cdot\! (D \, \nabla C) = 0
\]

for the tracer field \(C(x, y)\) with \(D = 10^{-10}\) m²/s by default
(small-molecule drug surrogate).  The two-step pattern is standard in
OpenFOAM and saves ~30% of the wall-clock cost compared with a coupled
scalar-momentum solver, while remaining exact for passive, non-reactive,
non-buoyant tracers.

## Boundary conditions

| Patch          | Momentum        | Scalar (T)           |
|----------------|-----------------|----------------------|
| `inlet_drug`   | fixedValue U_drug | fixedValue C = 1  |
| `inlet_medium` | fixedValue U_medium | fixedValue C = 0 |
| `outlet`       | zeroGradient    | zeroGradient         |
| `walls`        | noSlip          | zeroGradient         |
| `floor`        | noSlip          | zeroGradient (*)     |
| `frontAndBack` | empty (2D)      | empty (2D)           |

(*) Cell consumption is treated as a post-hoc sensitivity variable, not
baked into the forward solve — see Development Guide v2 §2.

## Flow split across inlets

`Q_drug = r_flow · Q_total`, `Q_medium = (1 − r_flow) · Q_total`.  The
inlet face areas depend on topology and are returned from
`generate_blockmesh_dict_v2` as `inlet_drug_area_m2` / `inlet_medium_area_m2`.

## Verification

`scripts/run_scalar_verification.py` sweeps four Peclet numbers
(`Pe ∈ {1, 10, 100, 1000}`) on a 1D channel and compares the simulated
`T(x)` against the analytic solution

\[
C(\xi) = \frac{1 - e^{-\mathrm{Pe}\,(1-\xi)}}{1 - e^{-\mathrm{Pe}}},
\qquad \xi = x / L
\]

with `C(0) = 1`, `C(L) = 0`.  The acceptance criterion is
`L2_rel_error < 2%` on a 100-cell mesh.

## API

::: ooc_optimizer.cfd.scalar.run_scalar_transport
::: ooc_optimizer.cfd.scalar.analytic_ad_1d
::: ooc_optimizer.cfd.scalar.run_scalar_verification_1d
