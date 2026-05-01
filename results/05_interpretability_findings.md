# 5. Interpretability Findings

If the project deliverable were *only* "the lowest L2 we could find", an aggressive grid scan plus a manual gradient-descent at the minimum would probably get within a few percent of the BO result, in less wall time. The pipeline earns its keep on four secondary outputs that grid search cannot produce: **Sobol total-effect indices, local-sensitivity gradients, fabrication-tolerance intervals, and the constraint-binding diagnostic.** This document explains what each output says, the F1/F2/F3 cross-topology spotlight findings, and how each piece answers a different practitioner question.

The formulas for all four outputs are in `02_methodology.md` §2.7. This document focuses on what the outputs *say* about the campaigns, not how they are computed.

---

## 1. Why interpretability lives on the two-inlet trio (mostly)

The bare-ladder optimisation is 2-D in `(W, Q_total)` by construction (`02_methodology.md` §2.2). With only two active dimensions, the Sobol/local/tolerance machinery has very little to chew on, and the ladder cannot meaningfully participate in *cross-topology* parameter comparisons (you cannot compare the Sobol weight of `r_flow` for the ladder, because `r_flow` is masked there).

The interpretability story is therefore much richer on the **four-active-dim two-inlet topologies** (`opposing` 5 dims, `same_side_Y` 4, `asymmetric_lumen` 4), and that is where the cross-topology spotlight findings (F1/F2/F3) come from. The ladder participates in two distinct interpretability narratives:

- The **2-D bare-ladder narrative** (`Q_total` dominant, ~ 87 % of `S_T`; `W` secondary at the AR-cap boundary).
- The **4-D pillar-ladder narrative** (`W` dominant, ~ 86 %; `Q_total` collapses to ~ 0; `s_p` is the secondary lever) — the regime-swap finding from `04_optimization_results.md` §3.

Section 2 below covers the cross-topology trio (F1/F2/F3); section 3 covers the surrogate-quality audit; section 4 covers the constraint-binding shift; section 5 covers fabrication tolerance.

---

## 2. F1 / F2 / F3 — the cross-topology spotlight findings

These three findings recur across all three two-inlet topologies and survive the surrogate-quality audit (§3). They are the publishable interpretability heuristics.

### F1. `r_flow` is the universal driver of the two-inlet class

| Topology | `S_T(r_flow)` | Local sensitivity rank for `r_flow` |
|---|---|---|
| `same_side_Y` | 0.860 | 1st (`|∂μ/∂x| = 1.27`) |
| `asymmetric_lumen` | 0.976 | 1st (`|∂μ/∂x| = 1.12`) |
| `opposing` | 0.651 (with caveat) | 2nd by rank, 1st by magnitude (`|∂μ/∂x| = 3.13`) |

`asymmetric_lumen` is essentially a **1-D problem in `r_flow`** — the surrogate is concentrated on a single dimension. `same_side_Y` is dominantly 1-D in `r_flow` with a small secondary effect from `W`.

**Mechanism.** `r_flow` is the only knob that moves the lateral position of the drug–medium interface, and the interface is the only structural feature these topologies can produce in the laminar regime. Wider `W` or higher `Q_total` can sharpen the interface but cannot move it.

**Implication for design-of-experiments transfer.** When a new lab adopts the pipeline, the smallest sufficient screen is **two BO runs on a single representative topology pair (one ladder-class, one two-inlet-class) at one chamber height**, then rely on the cross-topology heuristic to predict the dominant knob without rerunning Sobol on every new variant. This compresses the typical 5–10 BO runs per design study into a handful, lowering the entry cost.

### F2. `θ` (inlet convergence angle) is consistently inactive

| Topology | `S_T(θ)` | Local sensitivity for `θ` |
|---|---|---|
| `opposing` | 1.8 × 10⁻⁴ | floor noise |
| `same_side_Y` | 1.6 × 10⁻³ | floor noise |
| `asymmetric_lumen` | 3.5 × 10⁻⁵ | floor noise |

**Mechanism.** In the laminar regime sampled here (Re ≲ 50), the inlet jets relax to fully-developed channel flow within ~ 1 hydraulic diameter; the convergence angle `θ` only affects the inlet *transient*, which has dissipated long before the developed-flow region where L2 is computed. `θ` has no mechanical role in the long-channel response.

**Implication.** Future studies can fix `θ` at midpoint and save a dimension. This is a defensible search-space-pruning recommendation; we ran the campaigns with `θ` active to *demonstrate* it is inactive, not because we expected otherwise.

### F3. `W` and `Q_total` trade off as the secondary lever

After `r_flow` has placed the interface, the residual L2 reduction comes from chamber width (more diffusion length) **or** flow rate (higher Pe), depending on which limit is active for that topology. The two are interchangeable substitutes, not independent additives. Per-topology rank:

| Topology | `S_T(W)` | `S_T(Q_total)` | Secondary lever |
|---|---|---|---|
| `same_side_Y` | 0.13 | 0.077 | W |
| `asymmetric_lumen` | 0.013 | 0.033 | Q_total |
| `opposing` | 6.6 × 10⁻⁵ | 0.39 (with caveat) | Q_total |

**Mechanism.** Once `r_flow` has set the lateral interface position, the secondary L2 reduction comes from chamber width (more diffusion length) **or** flow rate (higher Pe), whichever knob is unconstrained in that topology's geometry. The two are substitutes, not additives.

### F4 (bonus). Flow-knob vs. interface-knob dichotomy across topology classes

The synthesis of F1 + F3 + the bare-ladder Sobol gives the cleanest single design heuristic the project produced:

| Topology class | Dominant Sobol parameter | Mechanism |
|---|---|---|
| **Imposed-inlet (ladder, christmas-tree)** | `Q_total` | gradient is *encoded into the boundary condition*; chamber's job is to advect the imposed pattern downstream without smearing. Dominant knob raises advective Pe → `Q_total`. |
| **Two-inlet coflow (opposing, SSY, asymmetric lumen)** | `r_flow` | gradient is *generated by the interface* between two streams; chamber's job is to *position* that interface → `r_flow`. |
| **Structured medium (ladder + 1×4 pillars)** | `W` | gradient depends on `W/s_p`; pillars decouple the chamber from a pure residence-time response → `W` becomes dominant. |

The dichotomy has no published precedent for tumor-on-chip gradient chambers. Its mechanistic content is independent of the specific dimensions or fluid properties used here, so it transfers to any prescribed-gradient design problem within the laminar regime.

---

## 3. Surrogate-quality audit (`Σ S_T` trustworthiness)

Sobol, local-sensitivity, and tolerance results are all claims about the *trained GP surrogate*, not direct claims about the underlying CFD response. The cheapest single-number self-check is `Σ S_T`: it equals 1 for a faithful surrogate (within Saltelli noise) and exceeds ~ 1.5 when the surrogate has overfit.

| Topology / config | H (μm) | Active dims | Σ S_1 | Σ S_T | Verdict |
|---|---|---|---|---|---|
| Ladder, no pillars | 200 | 2 | 0.989 | 1.014 | ✓ trustworthy |
| Ladder, no pillars | 300 | 2 | 0.989 | 1.013 | ✓ trustworthy |
| Asymmetric lumen | 200 | 4 | 0.965 | 1.022 | ✓ trustworthy |
| Same-side Y | 200 | 4 | 0.929 | 1.065 | ✓ trustworthy |
| Ladder, 1×4 pillars | 200 | 4 | 0.967 | 1.005 | ✓ trustworthy |
| Opposing | 200 | 5 | 0.701 | **1.813** | ⚠ overfit |

`opposing`'s `Σ S_T = 1.81` exceeds the 1.5 threshold. The high failure rate (39 %) compressed the GP into near-interpolation despite the noise floor we previously added. **Treat `opposing`'s `delta_W` dominance claim with caution; the magnitudes are inflated.** The directional content (`r_flow` and `delta_W` matter more than `Q_total`) is consistent with prior diagnostic findings, but the Sobol numbers themselves should not be reported as point estimates.

The other five surrogates are clean. The audit is itself a reportable methodological contribution: **publishing Sobol indices without a `Σ S_T` audit is publishing claims about a surrogate of unknown fidelity.** Future projects should adopt the audit as a default.

---

## 4. Constraint-binding diagnostic — the most actionable single output

For each of the five constraints, report at the BO optimum: the observed value, the threshold, and the slack. No formula; the value is descriptive but identifies which manufacturability or fluidic limit is binding — and therefore which lab-side capability would most cheaply move the optimum.

### 4.1 Why this is informative, not pathological

A naive reading of "the optimum is on three caps simultaneously" is that the BO has failed and is wedged in a corner because of bad bounds. The real interpretation is the opposite: **monotone response surfaces always pin at the corner of the feasible box.** Every Sobol total-effect index is non-zero with the same sign, the local gradients agree in direction, and the L2 surface is essentially flat near the corner — there is no interior minimum being missed. The corner is the optimum *under the constraints we chose to write down*.

Reading off which constraints bind (and which do not) is what tells the experimentalist where to invest:

- At H = 200 the binding pair is `(AR, τ)` → a thinner channel or a less shear-sensitive cell line moves the optimum.
- At H = 300 the binding pair is `(AR, Q_max)` → a higher-Q syringe pump or a wider chamber moves the optimum.

### 4.2 Constraint-binding tables — H = 200 vs H = 300 (ladder)

#### H = 200 ladder winner

| Constraint | Threshold | Observed | Slack | Verdict |
|---|---|---|---|---|
| `Re_max` | ≤ 100 | 24.97 | 75.0 | ✓ never binds; safety rail |
| `aspect_ratio_max` | ≤ 15 | 14.998 | 0.002 | **BINDING**; most active manufacturability constraint |
| `tau_mean_max` | ≤ 2.0 Pa | 1.998 | 0.002 | **BINDING**; at biology upper limit |
| `tau_mean_min` | ≥ 0.1 Pa | 1.998 | 1.898 | ✓ slack |
| `f_dead_max` | ≤ 0.08 | 0.031 | 0.049 | ✓ comfortable for ladder |

#### H = 300 ladder winner

| Constraint | Threshold | Observed | Slack | Verdict |
|---|---|---|---|---|
| `Re_max` | ≤ 100 | 41.71 | 58.3 | ✓ slack |
| `aspect_ratio_max` | ≤ 15 | 14.99 | 0.01 | **BINDING** |
| `tau_mean_max` | ≤ 2.0 Pa | 1.483 | 0.517 | ✓ released (74 % of cap) |
| `tau_mean_min` | ≥ 0.1 Pa | 1.483 | 1.383 | ✓ slack |
| `f_dead_max` | ≤ 0.08 | 0.021 | 0.059 | ✓ slack |
| `Q_total_max` (YAML) | ≤ 200 μL/min | 200.0 | 0.0 | **BINDING** (configuration choice, not physical) |

### 4.3 The H = 200 → H = 300 corner shift, summarised

Lifting H from 200 to 300 μm changes nothing about the geometry generator, mesh, or BO acquisition — only the τ-constraint slack. That single change relocates the binding pair from `(AR, τ)` to `(AR, Q_max)`, drops L2 by 17.9 %, and triples feasibility. **This is exactly the kind of single-knob design recommendation a constraint-aware BO is supposed to surface, and it would be invisible to a hand-tuned design or to a BO with no reported constraint-corner diagnostics.**

### 4.4 Concrete next-step implications from the H = 300 binding pair

Three concrete unlocks below 0.067:

- **Raise `Q_total_max` from 200 to 400 μL/min** (typical syringe-pump capacity ~ 1000). Expected L2 drop: another ~ 10 %. *Caveat:* τ scales with Q at fixed (W, H) and may re-bind the upper biology cap; safe ceiling is closer to Q ≈ 270 μL/min before τ rebinds.
- **Raise `W_max` to 6000 μm or add H = 400 μm.** Releases the AR/W corner. Within-strip step quantisation becomes the dominant L2 contribution somewhere around L2 ≈ 0.04 — beyond that, only the per-inlet `C_k` 8-D BO can help.
- **Open the constraint set** — allow W/H up to 18 (still within Folch's PDMS-stability bound). One-line YAML edit.

Detailed roadmap in `07_future_work.md`.

---

## 5. Fabrication-tolerance intervals — the implicit free lunch

The bisection routine returns the largest perturbation of each parameter (in physical units) that keeps L2 within +10 % of the optimum. The result is *much* looser than typical fabrication or operational precision — a free lunch the experimentalist would not otherwise know is on the table.

### 5.1 H = 200 ladder (interior optimum on Q)

| Param | Optimum | −Δ allowed | +Δ allowed | Tolerable range |
|---|---|---|---|---|
| `W` | 3000 μm | −1143 | +1501 | **[1857, 4500] μm** (~ ± 40 %) |
| `Q_total` | 119.5 μL/min | −42.8 | +80.2 | **[77, 200] μL/min** (~ ± 40 %) |

PDMS soft-litho precision is ± 5–10 μm; the design tolerates ± 540 μm. Syringe-pump precision is ± 1–2 %; the design tolerates ± 40 %. **The design is robust to fabrication and operational variance — does not require precision microfab.**

### 5.2 H = 300 ladder (corner optimum, asymmetric)

At H = 300 the optimum sits at the upper YAML bound for both active parameters (`W = 4496 ≈ 4500`, `Q_total = 200 ≈ 200`). The tolerance routine bisects within `[bound.min, bound.max]`:

| Param | Optimum | −Δ allowed | +Δ allowed |
|---|---|---|---|
| `W` | 4496 μm | −1675 | +4 (corner) |
| `Q_total` | 200 μL/min | −72.5 | 0 (corner) |

The `+Δ ≈ 0` is **not a numerical bug — it is direct evidence that the optimum has been pinned by the design-box ceiling**, not by the underlying CFD response surface. The asymmetry is itself an actionable signal: the experimentalist should *not* aim above the design point on either knob, but has substantial slack on the "smaller" side.

### 5.3 What each interpretability output answers, in three sentences

| Output | Question it answers |
|---|---|
| **Sobol `S_T`** | *What should the lab calibrate most carefully?* — the highest-`S_T` parameter is where small drift causes the most L2 degradation. |
| **Tolerance interval** | *How tight does fabrication / operational control have to be?* — gives the absolute drift budget per parameter. |
| **Constraint-binding** | *Which constraint should the lab try to relax to push performance?* — names the next investment that buys the next L2 unit. |

The three are complementary, not redundant. Sobol identifies QC priority. Tolerance bounds the QC slack. Constraint-binding identifies the next forward investment. Each is independently useful; the combination is the methodological deliverable.

---

## 6. Why the L2 surface is "monotone-into-the-corner" in this problem

A reviewer might reasonably ask: doesn't BO usually find an *interior* optimum? Why does this campaign produce corner-pinned winners on both ladder configurations?

The answer is mechanistic, not algorithmic:

- For the **bare ladder**, increasing `W` gives more cross-stream diffusion length per advective transit, smoothing the staircase steps. Increasing `Q_total` raises Pe_streamwise, sharpening the per-strip identity at the chamber outlet. **Both effects monotonically reduce L2** until the constraints stop the BO. The L2 surface is convex-into-the-corner with respect to the production constraint set; there is no interior minimum.
- For the **two-inlet topologies**, the interior of the response surface is essentially flat at the uniform-field floor (`L2 ≈ r_flow`-dependent). The BO finds the corner that minimises `|⟨C⟩_y(x) − target(x)|` after picking the `r_flow` that best places the interface — but since this is a fundamentally different problem (mass conservation forbids the target), the "corner" is just whichever combination of caps best approximates uniformity.

Corner-pinning is therefore the **physically expected behaviour** for the bare-ladder under our constraint set, not a BO failure mode. The pillar-ablation result confirms this interpretation: introducing pillars makes the chamber depart from a parallel-plate channel, breaks the monotone-in-Q response, and produces an *interior* optimum at `W ≈ 2100 μm` instead of the corner-pinned `W = 3000`.
