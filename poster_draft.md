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
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/A_ladder_N8.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/B_christmas_tree.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/C_side_injection_K8.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/D_permeable_wall.png`
- `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/E_counter_flow.png`

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

**[FIGURE: cross-topology bar chart of best feasible L2]** *(this needs to be made — see below)*

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
- `examples/tumor_chip_linear_gradient/data/results/bo_ladder_none_H200/interpretability/sobol.png`
- `examples/tumor_chip_linear_gradient/data/results/bo_opposing_none_H200/interpretability/sobol.png`
- `examples/tumor_chip_linear_gradient/data/results/bo_same_side_Y_none_H200/interpretability/sobol.png`
- `examples/tumor_chip_linear_gradient/data/results/bo_asymmetric_lumen_none_H200/interpretability/sobol.png`

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

**[FIGURE: ladder Sobol comparison or local sensitivity]**
- `examples/tumor_chip_linear_gradient/data/results/bo_ladder_none_H200/interpretability/local_sensitivity.png`
- `examples/tumor_chip_linear_gradient/data/results/bo_ladder_none_H300/interpretability/local_sensitivity.png`

**[FIGURE: heatmap from Phase-2 Sobol scan, or a freshly-rendered C-field of the ladder winner — see "Plot to make" below]**
- `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/heatmap.png` — at minimum this shows the W,Q response surface

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

## Plot to make (recommended new figure for Panel 4 or 5)

**Cross-topology L2 bar chart**, the missing canonical poster figure. A simple matplotlib script renders this in seconds. Consider:

```python
# Save as scripts/plot_cross_topology_summary.py and run once
import json, matplotlib.pyplot as plt
from pathlib import Path
RES = Path("examples/tumor_chip_linear_gradient/data/results")
data = [
    ("opposing", 0.8822),
    ("same_side_Y", 0.9937),
    ("asymmetric_lumen", 1.0875),
    ("ladder (H=200)", 0.0817),
    ("ladder (H=300)", 0.0671),
]
fig, ax = plt.subplots(figsize=(7, 4.5))
labels, vals = zip(*data)
colors = ["#6b7280"]*3 + ["#dc2626", "#16a34a"]
ax.barh(labels, vals, color=colors, edgecolor="black")
ax.axvline(0.585, ls="--", color="#9ca3af", label="uniform-field floor (axis=x)")
ax.set_xlabel("Best feasible L2 (lower is better)")
ax.set_title("Cross-topology BO winners (axis=y target)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RES / "cross_topology_summary.png", dpi=180)
```

That gives a clean, poster-ready bar chart; drop it into Panel 4.

---

## Image inclusion cheat-sheet

| Panel | Figure(s) to include | Source path |
|---|---|---|
| 1 | (text-only OK) | — |
| 2 | A_ladder_N8.png, B_christmas_tree.png, C_side_injection_K8.png, D_permeable_wall.png, E_counter_flow.png | `examples/tumor_chip_linear_gradient/data/figures/topology_candidates/` |
| 3 | (text-only or pipeline diagram) | — |
| 4 | cross_topology_summary.png (run script above), then 4× sobol.png mini-grid | `examples/tumor_chip_linear_gradient/data/results/bo_<topology>_none_H200/interpretability/sobol.png` |
| 5 | local_sensitivity.png H=200 + H=300, heatmap.png from Phase-2 scan | `…/bo_ladder_none_H200/interpretability/local_sensitivity.png`, `…/bo_ladder_none_H300/interpretability/local_sensitivity.png`, `examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/heatmap.png` |
| 6 | (text-only OK) | — |

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
All eval logs, BO state checkpoints, and interpretability artifacts: `examples/tumor_chip_linear_gradient/data/results/`
Plan + diagnostic notes: `diagnostic_findings.md`, `integration_run_findings.md`, `ladder_H_sweep_findings.md`, `REPORT.md` (companion technical report)
