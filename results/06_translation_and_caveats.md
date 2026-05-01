# 6. Translation to Bench, and Honest Caveats

This document closes the loop from CFD-result back to lab-experiment. Section 1 is the bench-ready chip spec for the H = 300 ladder winner. Section 2 captures the caveats that any reviewer or experimental partner will probe. Section 3 is the limits-of-the-present-analysis section — what this report's analysis does *not* establish, even given the cleaner H = 300 result.

---

## 1. Bench-ready chip — the H = 300 ladder winner

The H = 300 ladder winner is fully specified for fabrication.

### 1.1 Chip geometry

| Spec | Value |
|---|---|
| Chamber length L | 10 mm (fixed by problem statement) |
| Chamber width W | **4496 μm** (≈ 4.5 mm; AR = 14.99) |
| Chamber height H | **300 μm** |
| Number of inlet strips N | 8 |
| Per-strip inlet width | W/N = 562 μm |
| Strip layout | N strips at `x = 0`, stacked along `y` axis |
| Per-strip wall material | PDMS soft-lithography (one master, single-replica) |
| Aspect ratio W/H | 14.99 — at the edge of the safe range (literature consensus 10–20) |

### 1.2 Inlet-concentration ladder

The midpoint convention `C_k = (k+0.5)/N` for `k ∈ {0, ..., 7}`:

| Strip k | y-range (μm) | C_k (drug fraction) |
|---|---|---|
| 0 | [0, 562] | 0.0625 |
| 1 | [562, 1124] | 0.1875 |
| 2 | [1124, 1686] | 0.3125 |
| 3 | [1686, 2248] | 0.4375 |
| 4 | [2248, 2810] | 0.5625 |
| 5 | [2810, 3372] | 0.6875 |
| 6 | [3372, 3934] | 0.8125 |
| 7 | [3934, 4496] | 0.9375 |

The midpoint convention is **a free 38 % L2 reduction** over the more common endpoint convention `C_k = k/(N−1)`. It is the geometrically optimal choice for any imposed-inlet gradient generator (each strip's prescribed concentration matches the linear-target value at the strip's *centre*).

### 1.3 Flow specification

| Spec | Value |
|---|---|
| Total flow rate Q_total | **200 μL/min** |
| Per-strip flow rate | Q_total / N = 25 μL/min |
| Per-strip mean velocity U_x | Q_total / (W · H) = 0.247 mm/s = 2.47 × 10⁻⁴ m/s |
| Reynolds number Re | 41.71 (laminar; `Re_max = 100` cap not active) |
| Mean wall shear stress τ | 1.48 Pa (well below 2.0 Pa biology cap) |

### 1.4 Upstream hardware required

- **Eight pre-mixed reservoirs** at the eight `C_k` values, *or* a single binary mixer tree producing them on-chip (candidate B in the topology screen at `03_topology_screening.md` §3.3.2).
- **One precision syringe-pump pair** capable of 200 μL/min total, with ± 2 % accuracy. Sobol says this is the high-priority calibration target — pump precision affects L2 more than chamber dimensions do at this design point.
- **PDMS soft-lithography fabrication** at W/H = 15. The design tolerates ± 540 μm on W and ± 40 % on Q_total without losing more than 10 % L2 quality (see `05_interpretability_findings.md` §5).
- **Cell-line check.** At τ = 1.48 Pa sustained, the design is comfortable for endothelial cells, kidney/hepatocyte lines, and most cancer cell lines tested at literature shear levels. Sensitive primary cells (neurons, primary tumor explants) should run at H = 400 μm at the same W to drop τ further (predicted L2 ≈ 0.072 with a single-line YAML edit, still better than the H = 200 result).

### 1.5 Operational notes

The single most important note: **R² = 0.990 is achieved only at the design Q_total = 200 μL/min**. Halving Q drops Pe_streamwise and breaks streamline stratification, dropping R² noticeably. The chip is *specifically* a 200 μL/min linear-gradient generator. Calibrate Q with a flow-rate sensor, not just by syringe-pump nominal setting.

The chamber's **transit length** (the distance over which the imposed inlet ladder smooths to the linear target) is ~ 2 mm. Imaging stations downstream of `x ≈ 2 mm` see the converged gradient; upstream stations see the staircase shoulders. For dose-response readout, place the cells in the `x ∈ [3, 9] mm` range.

### 1.6 Suggested first experiment

**Fluorescent-tracer inflow at the eight C_k values, line-scan along y at three downstream stations** (`x = 1, 5, 9 mm`). Compares chip-vs-CFD agreement at three different stages of the transit. Predictions: at `x = 1 mm` the staircase shoulders should be visible (CFD shows the same); at `x = 5 mm` the field should be linear within ~ 1 % of the depth-averaged profile; at `x = 9 mm` essentially identical to `x = 5 mm`. Discrepancy between chip and CFD at any station is then attributable to a specific physical mechanism (inlet asymmetry, mesh diffusion, 3-D effects, etc.) and informs the next CFD refinement.

### 1.7 Pillar 1×4 variant — for shear-tolerant cell lines

The pillar-ablation winner offers an alternative chip spec with ~ 30 % lower L2 but higher τ:

| Spec | Pillar 1×4 winner | (vs H = 300 baseline above) |
|---|---|---|
| L2 | **0.0568** | 0.0671 |
| W | 2121 μm | 4496 μm |
| H | 200 μm | 300 μm |
| Q_total | 125 μL/min | 200 μL/min |
| Pillar diameter d_p | 184 μm | — |
| Pillar spacing s_p | 427 μm | — |
| τ_mean | 2.62 Pa (**violates production cap**) | 1.48 Pa |
| R²_lin | 0.992 | 0.990 |

Use the pillar variant only if the experiment's cell line tolerates τ ≈ 2.6 Pa over the full incubation. Otherwise the H = 300 baseline is the safer choice.

---

## 2. Honest caveats

These are publishable, reviewer-anticipated caveats. They are recorded here so that any consumer of the project's results knows what assumptions they are inheriting.

### 2.1 `opposing` Sobol indices have inflated magnitudes

`Σ S_T = 1.81` (above the 1.5 trustworthy threshold). Surrogate is overfit because the high failure rate (39 %) compressed the GP into near-interpolation. Report the directional content (`r_flow` and `delta_W` matter more than `Q_total`) but treat the magnitudes with caveat. The other five surrogates are clean (Σ S_T ∈ [1.005, 1.065]).

### 2.2 Two non-fatal `Field T uniform 0` extraction errors

Across the integration runs, ~ 1 % of CFD evaluations produced a degenerate concentration field (likely an extreme corner of the design space producing a mesh that `simpleFoam` survives but `scalarTransportFoam` doesn't). These are absorbed as penalty `L2 = 99` records, do not bias the BO, and were not investigated further. Frequency too low to matter.

### 2.3 Phase-2 (W, Q_total) Sobol scan looked "flat"

The 32-evaluation Sobol scan from the earlier diagnostic phase produced `L2 ∈ [0.172, 0.176]` — a 0.004 spread — apparently contradicting the production BO winner at L2 = 0.082. The discrepancy is fully explained by the **endpoint vs. midpoint inlet-concentration convention**: the scan used `C_k = k/(N−1)` (endpoint), production uses `C_k = (k+0.5)/N` (midpoint). The midpoint convention is uniformly 38 % better at any fixed geometry; that, plus the BO actively pushing W → AR cap and Q → τ cap, makes up the remaining factor of ~ 2.

### 2.4 Q_total is pinned at the YAML upper bound at H = 300

This means the *intrinsic* ladder L2 floor below 0.067 is unknown without raising the Q upper bound. 200 μL/min is a configuration choice, not a physical limit; typical syringe-pump capacity is ~ 1000 μL/min. **Worth noting in the writeup**: the 0.0671 number is the *constrained* ladder floor inside the production design box, not the topology's intrinsic floor.

### 2.5 Topologies B, C, D, E remain unimplemented

Each was screened with first-principles physical analysis but not run through CFD. Each has a "hidden parameter" set whose value determines whether the C-field is actually linear (vs. plateau, vs. saturating, vs. step-with-impingement-noise). Each requires non-trivial mesh and BC engineering; on the roadmap (see `07_future_work.md` Tier 3), not abandoned.

### 2.6 The pillar-ablation winner narrowly violates the τ cap

L2 = 0.0568 at τ = 2.62 Pa, above the production cap of 2.0 Pa. The best fully-feasible pillar design is L2 = 0.0588, comparable to the bare-ladder H = 200 winner. The pillar gain is real but only fully accessible under a relaxed τ cap or a less shear-sensitive cell line. We report both numbers; the constrained number is the cautious report.

---

## 3. Limits of the present analysis (what we do *not* claim)

These are the analysis's limits, distinct from the caveats above. Each one bounds the scope of the L2 = 0.0671 result.

- **Steady-state only.** All BO is performed against time-converged `simpleFoam` + `scalarTransportFoam` solutions. Transients during chip startup or reagent switching are not modelled. For dose-response experiments where the gradient is held steady, this is fine; for *time-varying* dose protocols the results do not transfer.
- **2-D laminar, fixed-H model.** The CFD is 2-D in xy with a fixed-H cell-thickness assumption. 3-D effects (top-and-bottom-wall boundary layers reducing effective free-stream Q, secondary flows in inlet manifolds) are not in this model. The 3-D validation module exists in the codebase (`ooc_optimizer/validation/cfd_3d_v2.py`) but was not run against the H = 300 winner in this cycle. The 2-D approximation is accurate for `W/H ≥ 5`, always satisfied here.
- **Drug-surrogate diffusivity.** `D = 10⁻¹⁰ m²/s` is appropriate for small molecules (~ 100–1000 Da). For larger biologics (peptides, antibodies, ~ 100 kDa) `D` drops to `10⁻¹¹–10⁻¹² m²/s`, raising Pe by 1–2 orders. The optimal Q_total for a high-Pe biologic gradient is *lower* than 200 μL/min, not higher — the present winner does not directly transfer. Re-running the BO with the new D is a one-line YAML edit.
- **Single target shape.** Only `linear_gradient` was actively optimised; `step` and `bimodal` targets remain TODO. The pipeline supports them, but the constraints (especially the AR cap at 15) may bind differently against a sharp step than against a smooth ramp.
- **No experimental validation.** The pipeline has not been validated against a fabricated chip. The R² = 0.990 is a CFD-vs-CFD-target metric; the chip-vs-CFD agreement is unknown until benchmark data exist.
- **Single cell-line model.** The τ ∈ [0.1, 2.0] Pa range is a generic compromise across cell lines. Sensitive primary cells (neurons, primary tumor explants) need a tighter cap (~ 1.0 Pa); tolerant immortalised lines can run at 2–5 Pa. The present winner is comfortable for tolerant lines but uncomfortable for sensitive ones at H = 200; **prefer H = 300 for sensitive cells** even before the L2 improvement is counted.

These limits are published as part of the result, not concealed. Each one suggests a distinct follow-up experiment. The tiered roadmap in `07_future_work.md` walks through them in order of cost / scientific yield.
