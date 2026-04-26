# Poster Draft — Tumor-on-Chip Inverse Design

> **How to use this file.** Each `### Panel N` block is a poster panel. Headings, bullets, and tables are paste-ready text. `**[FIGURE: ...]**` markers tell you which PNG to drop into the panel from the repository asset directories. Final layout is your call — this draft assumes a 6-panel landscape grid (3 columns × 2 rows) but content reflows easily.

---

## Title block

**Title.** Bayesian Optimization of Microfluidic Tumor-on-Chip Geometry for a Linear Drug-Concentration Gradient: A Topology-First Design Methodology

**Subtitle / one-sentence summary.** Mechanism-led pivot from two-inlet coflow to a Whitesides-style stacked-ladder topology, combined with a constrained CFD-surrogate Bayesian optimization, achieves a near-linear y-axis gradient (R² = 0.990) at L2 = 0.067 — a 9.4× improvement over the original axis=x design space.

**Authors / affiliation.** *(fill in)*

---

### Panel 1 — Motivation & Problem Statement

**Why a linear gradient?** A tumor-on-chip chamber that exposes 3D-cultured cells to a *spatially varying* drug concentration replaces a multi-chip dose curve with a single device — every cell sees a different dose simultaneously, enabling one-chip dose-response screening.

**The inverse-design problem.** Given a target concentration profile `C_target(x, y)`, find the chamber geometry + flow conditions that produce a CFD-simulated field closest to the target, subject to manufacturability and cell-biology constraints.

**Pipeline.** Parametric CAD → OpenFOAM (`simpleFoam` + `scalarTransportFoam`) → metric extraction (normalised-RMS L2, R²-to-linear, Re, Pe, τ, f_dead) → BoTorch BO with `ConstrainedExpectedImprovement` → per-topology Sobol sensitivity analysis on the trained GP.

**Initial finding.** A 600-evaluation campaign over three two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`) plateaued at L2 ≈ 0.63 — only 8% better than a uniform field, with monotonicity at chance level. **The L2 ceiling is not a BO failure; it is a geometric impossibility.**

**Mass-conservation argument** (the key insight). For steady incompressible flow with two coflow inlets and no through-wall scalar source, advective flux conservation across any vertical slice forces

$$\langle C \rangle_y(x) = r_{\text{flow}} \quad \forall x$$

A linear x-gradient requires `<C>_y(x) = x/L`, which varies with x. **Mathematically forbidden.** The natural achievable structure is a y-direction interface, not a streamwise gradient.

**[FIGURE: panel header diagram]** *(skip — text-only panel)*

---

### Panel 2 — Topology Screening (5 candidates)

**Five new topologies** were proposed and analysed from first principles (Pe, Re, mass conservation, manufacturability):

| | Topology | Target axis | Mechanism | Verdict |
|---|---|---|---|---|
| A | Y-stacked ladder | y | N inlet strips at x=0, prescribed C_k | ✓ Viable; prototyped |
| B | Christmas-tree mixer | y | Binary mixer tree → 8 chamber inlets | Future work |
| C | Distributed side-injection | **x** | K side ports along y=0 with x-varying Q_k | Future work |
| D | Permeable membrane wall | x | Graded-permeability floor + reservoir | Future work |
| E | Counter-flow inlets | x | Drug at x=0, medium at x=L, side outlets | Unsteady risk; not pursued |

**Key insight from screening.** A linear x-gradient requires a *distributed scalar source* (candidates C, D); the y-axis target is achievable with imposed-inlet topologies (A, B); counter-flow (E) violates steady-state. **Candidate A is cheapest to prototype and physically valid.**

**[FIGURE: 5 schematic renders, side-by-side]**
- `figures/01_topology_candidates/A_ladder_N8.png`
- `figures/01_topology_candidates/B_christmas_tree.png`
- `figures/01_topology_candidates/C_side_injection_K8.png`
- `figures/01_topology_candidates/D_permeable_wall.png`
- `figures/01_topology_candidates/E_counter_flow.png`

*Suggested arrangement: 2×3 mini-grid with E on its own underneath, since it's deprecated.*

---

### Panel 3 — Methods

**CFD model.** 2-D laminar single-cell-thick chamber, OpenFOAM v2406. `blockMesh` for parametric multi-block geometry; `simpleFoam` for momentum (frozen flow); `scalarTransportFoam` for passive-tracer transport on the converged velocity field. Diffusivity `D = 10⁻¹⁰ m²/s` (small-molecule drug surrogate). Diffusion is the only cross-stream mixing — no turbulence at our Re ≈ 0.1–40.

**Bayesian optimization.** BoTorch with Matérn-5/2 GP, Sobol-quasirandom initialisation (24 points), `ConstrainedExpectedImprovement` (Sobol-seeded L-BFGS multistart) over a `ModelListGP(obj, c1…c5)`. Objective: minimise normalised-RMS

$$L_2 = \frac{\sqrt{\frac{1}{N}\sum (C - C_{\text{target}})^2}}{\sqrt{\frac{1}{N}\sum C_{\text{target}}^2}}$$

**Constraint set** (5 GPs):

| Constraint | Threshold | Role |
|---|---|---|
| `tau_mean` | ∈ [0.1, 2.0] Pa | cell-viability shear window |
| `f_dead` | ≤ 0.08 | no stagnant pockets (`U < 0.1·U_mean`) |
| `Re` | ≤ 100 | laminar gate (safety rail) |
| `aspect_ratio` (W/H) | ≤ 15 | PDMS-collapse manufacturability |

**Diagnostic metrics added** to every evaluation: `Re`, `Pe_streamwise`, `Pe_crossstream`, `aspect_ratio`, `R²-to-linear`.

**Interpretability.** SALib Saltelli sampling (n=1024) on the trained GP for total/first-order Sobol indices; autograd-based local gradients at the optimum; bisection-based fabrication-tolerance intervals (10% L2 degradation budget).

**Ladder (winning topology) details.** N=8 inlet strips at x=0, each at prescribed concentration `C_k = (k+0.5)/N` (midpoint convention — strip centres land exactly on the linear target; a free 38% L2 improvement over the endpoint convention `C_k = k/(N-1)`). Per-inlet `U_x` uniform.

**[FIGURE: pipeline flowchart — text-only OK, or insert a custom diagram if you have one]**

---

### Panel 4 — Results: Cross-Topology BO Campaign

**4-topology BO campaign** at H = 200 μm, axis=y target, 200 evals/topology = 800 forward CFD solves; wall ≈ 1 h.

**[FIGURE: cross-topology bar chart of best feasible L2]**
- `figures/02_cross_topology_summary/cross_topology_summary.png`

| Topology | Best L2 | Feasible | Active dim |
|---|---|---|---|
| **`ladder`** (winner) | **0.082** | 89/200 (44%) | 2 (W, Q) |
| `opposing` | 0.882 | 122/200 (61%) | 5 |
| `same_side_Y` | 0.994 | 127/200 (64%) | 4 |
| `asymmetric_lumen` | 1.088 | 47/200 (24%) | 4 |

**Cross-topology Sobol indices** (the publishable design heuristic):

| Topology | Dominant param | S_T | Subdominant | S_T |
|---|---|---|---|---|
| `ladder` | **`Q_total`** | 0.87 | `W` | 0.14 |
| `same_side_Y` | `r_flow` | 0.86 | `W` | 0.13 |
| `asymmetric_lumen` | `r_flow` | 0.98 | `Q_total` | 0.03 |
| `opposing` | `delta_W` | 0.78 | `r_flow` | 0.65 |

**Headline heuristic.** *Ladder is dominated by a flow knob (`Q_total`); two-inlet topologies are dominated by an interface-position knob (`r_flow`).* These are fundamentally different control regimes for prescribed-gradient design.

**[FIGURE: per-topology Sobol bars]**
- `figures/03_sobol_per_topology/sobol_ladder_H200.png`
- `figures/03_sobol_per_topology/sobol_opposing_H200.png`
- `figures/03_sobol_per_topology/sobol_same_side_Y_H200.png`
- `figures/03_sobol_per_topology/sobol_asymmetric_lumen_H200.png`

*Suggested arrangement: 2×2 mini-grid.*

---

### Panel 5 — Results: H-Sweep & Constraint-Corner Shift

**Extended ladder-only BO** at H = 200 vs H = 300 μm, 200 evals each, ~13 min wall. Goal: open the `aspect_ratio_max=15` cap from W ≤ 3000 μm to W ≤ 4500 μm.

| H (μm) | Best L2 | Best W (μm) | Best Q (μL/min) | τ at opt (Pa) | AR at opt | R² | Feasibility |
|---|---|---|---|---|---|---|---|
| 200 | 0.0817 | 2999 (cap) | 119.8 | **1.998 (cap)** | **15.00 (cap)** | 0.987 | 37% |
| **300** | **0.0671** | **4496 (cap)** | **200.0 (cap)** | 1.483 | **14.99 (cap)** | **0.990** | **96.5%** |

**Δ vs H=200: −17.9% L2, +0.3 pp R², +160% feasibility rate.**

**Constraint-corner shift** (the key mechanistic finding):
- H=200 corner: bound on `aspect_ratio_max` AND `tau_mean_max`.
- H=300 corner: bound on `aspect_ratio_max` AND `Q_total_max` (the YAML upper bound, not a physical limit). **`tau_mean` releases** because `τ ∝ Q/(W·H²)` — doubling-and-a-half H gives 2.25× tau headroom that BO immediately spends on higher Q.

**Feasibility leap (37% → 96.5%)** is a major collateral benefit beyond L2: the H=300 GP surrogate sees ~2.6× more useful CFD evaluations at the same compute budget, so its Sobol/sensitivity outputs are higher-quality.

**[FIGURE: ladder local sensitivity, H=200 vs H=300]**
- `figures/04_local_sensitivity/local_sensitivity_ladder_H200.png`
- `figures/04_local_sensitivity/local_sensitivity_ladder_H300.png`

**[FIGURE: ladder fabrication-tolerance intervals, H=200 vs H=300]**
- `figures/05_tolerance/tolerance_ladder_H200.png`
- `figures/05_tolerance/tolerance_ladder_H300.png`

**[FIGURE: Phase-2 Sobol scan over (W, Q_total) — design-space response surface]**
- `figures/06_phase2_scan/phase2_W_Q_scan.png`

**Top-10 ladder candidates form a tight cluster** at both H values:
- H=200: W ∈ [2987, 3000], Q ∈ [117.9, 119.5], L2 ∈ [0.0818, 0.0820], R² ∈ [0.987, 0.988].
- H=300: W ≈ 4490, Q ≈ 200, L2 ≈ 0.067, R² ≈ 0.990.

**Tolerance intervals at H=300** (10% L2 budget):
- `W` allowed range: [2821, 4500] μm — ~±18%, far beyond PDMS soft-litho ±5–10 μm precision.
- `Q_total` allowed range: [128, 200] μL/min — far beyond syringe-pump ±1–2% precision.
- **Design is robust to fabrication and operational variance.**

---

### Panel 6 — Discussion & Future Work

**The L2 stack — topology vs. optimization.**

| Stage | L2 | Multiplicative gain |
|---|---|---|
| Original 3 topologies + axis=x BO | 0.6343 | 1.0× (baseline) |
| Topology pivot to ladder (axis=y, hand-picked W,Q) | 0.110 | **5.8×** ← topology |
| H=200 ladder + 2-D BO over (W, Q) | 0.0818 | × 1.34 |
| H=300 ladder + 2-D BO + relaxed AR cap | **0.0671** | × 1.22 |
| (Projected) per-inlet `C_k` 8-D BO | ~0.04 | × ~1.7 |

**Topology selection earns ≈6× — the largest single jump.** Each subsequent BO/constraint-relaxation layer adds 1.2–1.3×. The BO and Sobol layers also produce the **fabrication-tolerance, dominant-parameter ranking, and constraint-binding diagnostics** that no hand-picked design could deliver.

**Manufacturability + biology assessment.**
- `aspect_ratio = 15` is the most active manufacturability constraint at both H values; literature consensus is W/H ≤ 10–20, so we are at the relaxed edge of the safe range.
- `tau_mean = 1.99 Pa` at the H=200 winner is at the upper biology limit; sensitive cell lines (primary tumor, neurons) may stress. **The H=300 winner at 1.48 Pa is comfortable for most cell lines** — another reason to prefer H=300 even before the L2 improvement is counted.
- `Re_max ≤ 100` never binds (max observed: 41.7 across 1200 evals); functions as a safety rail.

**Honest caveats.**
- `opposing` Sobol indices have ΣS_T = 1.81 (above the 1.5 trustworthy threshold) — high failure rate compressed the GP into near-interpolation; magnitudes are inflated.
- The H=300 winner has `Q_total` pinned at the YAML upper bound (200 μL/min); the *intrinsic* ladder L2 floor below 0.067 is unknown without raising that bound.
- Topologies B–E remain prototyped only as schematic figures; each requires 1–1.5 days of additional mesh/BC engineering.

**Novelty.** Christmas-tree + Kriging surrogate is published prior art (Yang 2020, Hashemi-Tilehnoee 2025). **The combination of (i) tumor-on-chip 3D-culture chamber, (ii) cross-topology Sobol sensitivity comparison on CFD-trained GPs, (iii) explicit fabrication-tolerance reporting derived from the surrogate, and (iv) the dimensional-physics constraint set wired into the BO acquisition is novel.**

**What's next** (in priority order):
1. **Open the search box** (`Q_total_max → 400`, add H = 400 μm). YAML-only, no code. Resolves whether L2 < 0.067 is tractable just by relaxing bounds.
2. **Per-inlet `C_k` 8-D BO** at fixed (W=4500, H=300, Q=Q_max). Lets the BO absorb cross-stream-diffusion bias by tuning inlet concentrations slightly off the linear ladder. Expected floor ~0.04. ~1 afternoon implementation cost.
3. Mesh refinement → cell-line-specific reruns → topologies B–E → density-based topology optimization (Borrvall–Petersson 2003).

---

## Image inclusion cheat-sheet (all paths relative to this `poster/` folder)

| Panel | Figure(s) | Path |
|---|---|---|
| 1 | (text-only) | — |
| 2 | 5 topology schematics | `figures/01_topology_candidates/{A_ladder_N8, B_christmas_tree, C_side_injection_K8, D_permeable_wall, E_counter_flow}.png` |
| 3 | (text-only or pipeline diagram) | — |
| 4 | Cross-topology bar chart | `figures/02_cross_topology_summary/cross_topology_summary.png` |
| 4 | 4× per-topology Sobol grid (H=200) | `figures/03_sobol_per_topology/sobol_{ladder,opposing,same_side_Y,asymmetric_lumen}_H200.png` |
| 5 | Local sensitivity comparison | `figures/04_local_sensitivity/local_sensitivity_ladder_{H200,H300}.png` |
| 5 | Fabrication tolerance comparison | `figures/05_tolerance/tolerance_ladder_{H200,H300}.png` |
| 5 | Phase-2 (W, Q) scan heatmap | `figures/06_phase2_scan/phase2_W_Q_scan.png` |
| 5/6 | Sobol after H lift | `figures/03_sobol_per_topology/sobol_ladder_H300.png` |
| 6 | (text-only) | — |

## Plot interpretation notes (Q&A from poster review)

### Q1 — How do you read the cross-topology bar chart (Panel 4)?

The dashed grey line at L2 ≈ 0.585 is the **uniform-field floor for the original axis=x target** — the L2 a perfectly flat C field would score against a linear x-ramp, derived from the mass-conservation argument in Panel 1. Three points to take away:

1. **The three two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`) sit at L2 ∈ [0.88, 1.09].** Worse than the uniform-field floor for axis=x because the target is now axis=y — the same mass-conservation argument that capped them on x still stops them on y. They are *fundamentally the wrong topology class* for any cardinal-axis linear gradient inside this chamber, not just axis=x.
2. **Both ladder bars (red, green) sit ~10× lower at L2 ∈ [0.067, 0.082].** That gap is what "topology change" buys — far more than any single optimisation hyperparameter or constraint relaxation can deliver.
3. **The H=200 → H=300 step (red → green) is small in absolute L2 (~0.015) but represents a 17.9% relative drop and 2.6× higher feasibility rate.** Topology selection earns the order-of-magnitude jump; H tuning earns the last accessible margin within manufacturability bounds.

The chart is a deliberate visual rebuttal to "the BO failed" — the BO did exactly what BO does; the *design space* was wrong.

### Q2 — What happened to the +x axis of the H=300 tolerance plot (Panel 5)?

`tolerance_ladder_H300.png` is **single-sided** (only −x extent). Both bars terminate at 0 on the right because **the H=300 winner sits at the upper YAML bound for both active parameters**:
- `W = 4495.6 μm` ↔ `W_max = 4500 μm` (and AR=15 cap, geometrically equivalent at H=300).
- `Q_total = 200.0 μL/min` ↔ `Q_max = 200 μL/min`.

The tolerance routine bisects within `[bound.min, bound.max]` searching for the largest perturbation that keeps L2 within +10% of the optimum. With the optimum *on* the upper bound, the +x search starts at the bound and immediately exits — there is no room to perturb upward. The asymmetry is therefore not a numerical bug, it is **direct evidence that the optimum has been pinned by the design-box ceiling**, not by the underlying CFD response surface.

Compare with `tolerance_ladder_H200.png`, which is **two-sided** for `W` (≈±18%) but lopsided for `Q` (small +x extent, larger −x). At H=200 the optimum sits on the AR cap (W=3000) and within Q range — so W tolerance is one-sided, Q is two-sided.

### Q3 — Three constraint caps simultaneously: what does it reveal?

The H=300 winner is pinned at **three caps at once**:
1. `W = 4496 ≈ W_max = 4500 μm` (YAML upper bound).
2. `aspect_ratio = 14.99 ≈ AR_max = 15` (manufacturability constraint; geometrically degenerate with W_max at H=300 since W/H = 4500/300 = 15).
3. `Q_total = 200.0 ≈ Q_max = 200 μL/min` (YAML upper bound).

So #1 and #2 are **the same cap viewed two ways**, leaving two genuine corner-binding directions: max-W and max-Q. Three observations:

**(a) The L2-decreasing direction is mechanistic.** Larger W gives more cross-stream diffusion length per advective transit time → smoother stair-step → lower L2. Larger Q raises Pe_streamwise → less axial smearing → cleaner per-strip identity at the chamber outlet. Both effects push toward a more linear field, so the BO walks both knobs to their bounds. **The optimum is corner-pinned because the response surface is monotone in both directions inside the box** — this is *why* the bar chart's H=200 vs H=300 difference is so clean.

**(b) Sobol S_T and the corner are consistent.** `Q_total` carries 87% of total-effect Sobol variance and `W` carries 14% — together ≈100%, which means the GP surrogate is an essentially 2-D function in (W, Q). Both the Sobol indices and the corner-pinning say the same thing: the response surface is monotone-decreasing in both, with Q dominant. The corner is not an artefact of GP overfit; it is the GP correctly reporting "go further if you can."

**(c) Practical interpretation: the reported L2 = 0.0671 is an *upper bound on what this topology can achieve*, not its intrinsic floor.** Three concrete unlocks:
- **Raise `Q_total_max` to 400 μL/min** — release the Q corner. Expected L2 drop: another ~10%, because Pe_streamwise scales linearly in Q. *Caveat:* tau_mean scales with Q at fixed (W, H) and may re-bind the upper biology cap before L2 saturates. Tau headroom at the H=300 winner is 1.48 → 2.0 Pa = 35%, so doubling Q would breach the cap; the actual safe ceiling is closer to Q ≈ 270 μL/min.
- **Raise `W_max` to 6000 μm or add H = 400 μm** — release the AR/W corner. The intrinsic ladder floor at fixed W/H = 15 should keep dropping with W until cross-stream diffusion fully bridges adjacent strips. *Caveat:* the within-strip step quantisation (the C field is a smoothed staircase) becomes the dominant L2 contribution somewhere around L2 ≈ 0.04 — beyond that, only the per-inlet `C_k` 8-D BO can help.
- **Open the constraint set** — allow W/H up to 18 (still within Folch's PDMS-stability bound). One-line YAML edit.

**(d) Methodological lesson for the paper.** Constraint-corner pinning at the optimum is itself a *publishable design heuristic* and not a failure mode. It tells the experimentalist exactly which manufacturability or fluidic limit is binding — i.e., which lab-side capability would most cheaply move the optimum. In our case the binding pair shifts from (AR, tau) at H=200 to (AR, Q_max) at H=300, telegraphing the next investment: a higher-Q syringe pump unlocks more L2 headroom than thinner channels.

---

## Numbers to memorise for the elevator pitch

- **L2 = 0.0671** at the H=300 ladder winner.
- **R² = 0.990** to a linear y-fit.
- **9.4× improvement** over the original axis=x BO winner (0.6343 → 0.0671).
- **800 + 400 = 1200 CFD evaluations** total across the full campaign.
- **96.5% feasibility rate** at H=300 vs 37% at H=200.
- **Q_total carries 87% of total-effect Sobol variance** — the dominant control knob for ladder.
- Fabrication tolerance on W: **±18% at H=300**, ~1000× the PDMS soft-litho precision.

## Acknowledgements / data availability

Code: `https://github.com/<org>/ooc_loop` *(fill in)*
Companion technical report: `REPORT.md` (in this folder)
Phase findings: `findings/diagnostic_findings.md`, `findings/integration_run_findings.md`, `findings/ladder_H_sweep_findings.md` (in this folder)
Full eval logs and BO state checkpoints (not bundled — too large): `examples/tumor_chip_linear_gradient/data/results/` in the source repository
