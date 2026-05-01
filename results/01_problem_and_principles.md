# 1. Problem and First Principles

This document explains *what* the project is trying to do and *why* the methodological choices follow from first principles. It is intentionally pre-results: nothing here depends on having run any campaign. A reader who follows §3 should be able to predict the diagnostic-phase failure of two-inlet topologies before seeing any L2 numbers.

---

## 1.1 Problem statement

A microfluidic tumor-on-chip (ToC) chamber is an *in vitro* platform that integrates living cells (typically 3D-cultured) within a precisely controlled microfluidic geometry. We want a chamber that exposes the cells to a *spatially varying* concentration of a soluble drug (or surrogate tracer) so that every cell sees a different dose simultaneously. The flagship target is a **linear concentration gradient across the cell-culture region**, replacing a multi-chip dose-response assay with a single device.

Formally, given a target field `C_target(x, y)` over the chamber and a parametric chamber + flow design vector `x ∈ X` (geometry, total flow, etc.), find

$$x^* = \arg\min_{x \in X_{\text{feas}}} \; L_2\bigl(C_{\text{CFD}}(x), C_{\text{target}}\bigr) \quad \text{where} \quad L_2 = \frac{\|C_{\text{CFD}} - C_{\text{target}}\|_2}{\|C_{\text{target}}\|_2}.$$

The feasible set `X_{feas}` encodes the biological and manufacturability constraints (Section 2). `C_CFD(x)` is the CFD-simulated concentration field for the design `x`, computed by solving the steady incompressible Navier–Stokes equations followed by passive scalar transport on the converged velocity field.

This is an **inverse-design problem**: the parameters that enter `C_CFD` are the optimisation variables, and CFD is the black-box that maps each candidate `x` to an evaluation of the L2 objective.

## 1.2 Why a linear gradient (and why "linear-in-y" was a non-trivial choice)

A linear gradient is the simplest spatially varying target compatible with a one-chip dose-response readout: each cell's position maps monotonically to a unique concentration, so the cellular response can be regressed against position to produce a dose-response curve from a single device. The original target was **linear in `x` (the flow direction)**, motivated by reading the gradient out along the chamber length in standard microscope frames.

The diagnostic phase (§3 below) showed that linear-in-`x` is *forbidden* in two-inlet coflow under the project's biological/manufacturability constraints. The target was therefore reformulated as **linear in `y` (the cross-stream direction)**, which the imposed-inlet ladder topology can produce by construction. This change is not a relaxation of the science: a transverse gradient is read out by line-scanning along `y` instead of along `x`, and it requires the same single chip and the same total reagent volume.

## 1.3 Why we use CFD-in-the-loop instead of a closed-form model

Two reasons. First, the constraints we care about (`tau_mean`, `f_dead`, `Re`, `W/H`) are by-products of the velocity field, not of the geometry alone — there is no closed-form mapping from the design vector to those quantities for any non-trivial chamber. Second, the response surface of `L2` over `(W, H, Q_total, ...)` is *not* monotone for the general cross-stream-mixed field; even when it is monotone (the bare-ladder case turns out to be), monotonicity is a finding, not an assumption. CFD is the cheapest faithful evaluator for both the objective and the constraints.

The CFD itself is conventional: 2-D laminar, OpenFOAM v2406, `simpleFoam` for momentum, `scalarTransportFoam` for the passive tracer on the frozen velocity field. The diffusivity is `D = 10⁻¹⁰ m²/s` (small-molecule drug surrogate, ~100–1000 Da). At the chamber's operating point (Re ≈ 0.1–40, Pe ≈ 10⁴–10⁷) the regime is laminar, advection-dominated, and steady — there is no turbulence, and any single CFD evaluation costs ~12 s wall on the project's hardware.

## 1.4 Why a Bayesian-optimization loop

Three properties of the problem make BO the right choice:

1. **Each evaluation is expensive** (~12 s of CFD per design). 200 evaluations is a full afternoon; we cannot afford grid scan or evolutionary algorithms with population-level exploration.
2. **The constraints are hard, non-convex, and not given in closed form.** Each constraint must be modelled as a separate black-box. We use one Gaussian-process (GP) surrogate per constraint and aggregate via `ConstrainedExpectedImprovement` so the acquisition can balance objective improvement against the constraint feasibility probability (`bo_loop.py`).
3. **The trained GP surrogates are reusable for interpretability.** Once BO converges, the same surrogate that drove the search is queryable for sensitivity (Sobol total-effect indices), local gradients (autograd through the GP mean), fabrication-tolerance intervals (bisection on the GP mean), and constraint-binding diagnostics. **The optimization is the engine; the interpretability triple is the deliverable.**

## 1.5 The mass-conservation principle (the project's pivotal insight)

This is the single most important mechanistic argument in the project. It applies to *any* prescribed-field design problem in the laminar regime with no through-wall scalar source.

**Setup.** Consider steady incompressible flow in a 2-D rectangular chamber of length `L` (along `x`) and width `W` (along `y`). Two inlets at `x = 0` carry the drug stream (concentration `C = 1`) and the medium stream (`C = 0`) at fractional flow rates `r_flow` and `1 − r_flow` of the total volumetric flow `Q_total`. The four side walls (`y = 0, W`) and the top/bottom (in 3-D, the floor and ceiling at `z = 0, H`) are no-flux for both momentum (no-slip) and scalar (`∂C/∂n = 0`). The single outlet is at `x = L`. The scalar obeys the steady advection–diffusion equation

$$\nabla \cdot (\mathbf{u}\,C - D\,\nabla C) = 0.$$

**Argument.** Define the depth-averaged concentration `⟨C⟩_y(x) = (1/W) ∫_0^W C(x, y)\,dy`. Integrate the advection–diffusion equation across the chamber width. The transverse diffusive flux `−D ∂_y C` vanishes at `y = 0` and `y = W` by no-flux boundary conditions, so

$$\partial_x\!\left[\langle u_x C\rangle_y(x)\right] = 0.$$

Streamwise averaging mass conservation across the chamber forces `⟨u_x C⟩_y` to be invariant in `x`. For a long shallow channel `u_x ≈ Q_total / (W H)` is approximately uniform in `y`, so

$$\langle C\rangle_y(x) \approx r_{\text{flow}} \quad \forall x \in [0, L].$$

(The inlet mixing-cup average is `(c_low (1 − r_flow) + c_high r_flow) = r_flow` for `c_low = 0`, `c_high = 1`.)

**Implication.** A linear-`x` target requires `⟨C_target⟩_y(x) = x/L`, which varies with `x`. **No combination of the seven design parameters can produce that** in two-inlet coflow because there is no source or sink of scalar mass between `x = 0` and `x = L`. The L2 floor at `≈ 0.585` is the geometric lower bound for any uniform field at `C = r_flow` against the linear target — no design can break it.

**The closest a BO can get** is "make `C(x, y)` as uniform as possible at `C = r_flow`" — which is exactly what the diagnostic-phase BO did (winner `opposing` at L2 = 0.6343, only 8% above the floor).

## 1.6 The admissibility table — turning the principle into a pre-screen

The mass-conservation argument generalises to a small admissibility table that any subsequent project can reuse before running CFD. ✓ = the topology class can in principle produce that target shape; ✗ = mass conservation forbids it; — = not applicable.

| Target shape | Two-inlet coflow (opposing, SSY, asym. lumen) | Imposed-inlet ladder (axis = y) | Distributed source (side injection, permeable wall) | Counter-flow (axis = x) |
|---|---|---|---|---|
| Linear in `x` | **✗** | — | ✓ | ✓ (steady-state conditional on Re) |
| Linear in `y` | ✓ (limited; depth-averaged value pinned at `r_flow`) | **✓** | — | ✗ |
| Step in `y` | ✓ | ✓ | ✓ | unsteady |
| Bimodal in `y` | ✗ | ✓ | ✓ | unsteady |

The pre-screen costs zero CFD evaluations and would have saved the diagnostic-phase 600-evaluation campaign.

## 1.7 Why "topology-first"

The mass-conservation principle implies that the topology class fixes a *floor* on achievable L2; parameter optimization can only move within the box that the topology imposes. In our L2 stack:

| Stage | Best L2 | Multiplicative gain |
|---|---|---|
| Original 3 topologies + axis-`x` BO | 0.6343 | 1.0× (baseline) |
| Topology pivot to ladder + axis-`y`, hand-picked geometry | 0.110 | **5.8×** ← topology |
| H = 200 ladder + 2-D BO over (W, Q_total) | 0.0818 | × 1.34 |
| H = 300 ladder + 2-D BO + relaxed AR cap | 0.0671 | × 1.22 |
| 1×4 pillar ablation, constraint-relaxed | 0.0568 | × 1.18 |

**Topology selection is the largest single jump (≈6×).** The BO and constraint-relaxation layers earn 1.2–1.3× each, but only after the topology floor has been lifted. This is the empirical justification for "topology-first": parameter tuning on the wrong topology is a category error.

## 1.8 Constraint set — biology, manufacturability, safety

The five constraints encode three categories of real-world limits:

| Constraint | Threshold | Category | Justification |
|---|---|---|---|
| `tau_mean` | ∈ [0.1, 2.0] Pa | biology | Cells stress above ~2 Pa over multi-hour experiments; below ~0.1 Pa, advective transport is too weak to deliver fresh medium. |
| `f_dead` | ≤ 0.08 | biology / flow | Fraction of the developed-flow region with `U < 0.1·U_mean`. Stagnant pockets accumulate dead cells and reagent. |
| `Re` | ≤ 100 | safety | Laminar gate. Above ~100, the steady-state CFD assumption itself becomes questionable. |
| `aspect_ratio` | `W/H ≤ 15` | manufacturability | PDMS-channel collapse during plasma bonding for shallow, wide channels (Folch lab consensus 10–20; we chose 15 for headroom). |
| `mesh_ok ∧ converged` | binary | numerical | The CFD must run to completion and the residuals must drop. |

These are encoded as five independent GP constraint surrogates fed into `ConstrainedExpectedImprovement`. The choice is not arbitrary: every constraint corresponds to a measurable lab quantity and a documented failure mode. The justification for each in 1–2 sentences is the *constraint-set rationale* that any reviewer will probe.

## 1.9 The interpretability triple

Beyond locating an optimum, the trained GP surrogate yields three *post-hoc* analyses (full formulas in `02_methodology.md` §3.5):

- **Sobol total-effect indices `S_T,i`** (`ooc_optimizer/analysis/sobol.py`) — what fraction of the variance in the GP-modelled objective is explained by parameter `i`, including all interactions. Computed by SALib Saltelli sampling at n = 1024 on the trained surrogate.
- **Local sensitivity `|∂μ/∂x_norm|`** at the BO optimum — the magnitude of the GP-mean gradient with respect to unit-cube-normalised inputs. Computed by autograd through the trained GP.
- **Bisection-based fabrication-tolerance intervals** — the largest perturbation of each parameter that keeps `μ(x* + Δx_i e_i) ≤ 1.10 · μ(x*)`, with all other dimensions fixed. The bisection runs on the GP mean (no extra CFD calls).

A fourth artefact, the **constraint-binding diagnostic**, simply lists which constraint is active at the optimum and how much slack the others have. This is mechanically trivial but turns out to be the most actionable single output of the entire pipeline (`05_interpretability_findings.md` §4).

## 1.10 What is genuinely novel

The literature survey (Yang 2020, Hashemi-Tilehnoee 2025, Whitesides 2000, Dertinger 2001, Ayuso 2020, Borrvall–Petersson 2003) confirms that:

- **Christmas-tree + Kriging surrogate** for general microfluidic mixers is published prior art.
- **Bayesian optimization for general microfluidics** (Kundacina 2025) is published prior art.
- **Sobol sensitivity analysis on a CFD-trained GP surrogate of a tumor-on-chip linear-gradient chamber** has no published precedent.

The defensible novelty bundle is therefore:

1. The **mass-conservation pre-screen** as a generalisable feasibility test for any prescribed-field laminar-microfluidic design problem.
2. The **constraint-aware BO with five GP constraint surrogates** and a dimensional-physics diagnostic-metric set (Re, Pe, AR, R²-to-linear).
3. The **post-hoc interpretability triple** that turns the trained surrogate into Sobol + local-sensitivity + tolerance + binding outputs.
4. The application of (1)–(3) to a **3D-culture tumor-on-chip chamber** specifically — bridging the design-of-experiments literature and the laminar-mixer literature.

The BO loop and the ladder topology are well-engineered scaffolding around these four contributions; they are not novel in isolation.
