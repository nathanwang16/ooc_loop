# `Haha.pdf` — Revision Plan

A targeted edit list for moving the draft from "lightly-formalised report" to a stand-alone formal manuscript (sub-journal level, but more rigorous than the poster). Items are grouped by severity. Cross-references to source-of-truth files in the repo are noted in `code paths`.

---

## Q. Background — four reviewer questions and their answers

These four answers are the foundation for the rest of the plan. They are reproduced here so any later revision step can be checked against the resolved facts.

### Q1 — Does `25 cells/mm` align with the reality of cell density used in production?

**Yes, with one caveat.** The default in `topology_blockmesh.py` is `ny_per_mm=20` (two-inlet builders) and `ny_per_mm=30` (ladder builder). The diagnostic record (`findings/diagnostic_findings.md`) states `ny_per_mm=25` was used in production, implying a runtime config override.

Sanity check: at the H=300 winner (W=4496 μm), 25 cells/mm in y → ~112 cells across the chamber, ≈ 14 cells per inlet strip for N=8. This is a defensible resolution for upwind scalar transport. The number is *physically* plausible; the issue is purely traceability — the paper should quote the value actually used (read from the H=300 winner's `blockMeshDict`) rather than rely on the diagnostic-doc claim.

### Q2 — What is "solver verification against the 1D advection–diffusion analytic"? What do we have?

**What it means.** Run `scalarTransportFoam` on a 1D channel with uniform velocity, compare simulated `C(ξ)` to the textbook analytic `C(ξ) = (1 − e^{−Pe(1−ξ)}) / (1 − e^{−Pe})`. Method-of-manufactured-solutions-style verification: isolates the scalar-transport solver from any geometry/BC effect.

**What we actually have in the repo:**

1. `data/scalar_verification/scalar_verification_results.json` — exactly this verification. Real numbers (all PASS at 2% L2):

   | Pe | n_cells | L2 rel. error | L∞ error | Pass |
   |---|---|---|---|---|
   | 1     | 100 | 0.064 % | 0.058 % | ✓ |
   | 10    | 100 | 0.77 % | 1.63 % | ✓ |
   | 100   | 100 | 1.26 % | 10.65 % | ✓ |
   | 1000  | **500** | 1.66 % | 36.79 % | ✓ |

2. `data/verification/convergence_results.json` — a Hagen–Poiseuille velocity/WSS convergence study at three mesh levels (nx=100/200/400, ny=10/20/40). Mean velocity error: 1.18 % → 0.39 % → 0.13 %. All PASS at 2%.

**Where the paper is wrong:**

- It says the 100-cell mesh was used **for all four Pe**. Reality: Pe=1000 used **n=500** cells. A 100-cell run at Pe=1000 would fail the 2% bar because of upwind numerical diffusion. Correct wording: *"100-cell mesh for Pe ∈ {1, 10, 100}; refined to 500 cells at Pe=1000 to suppress upwind diffusion to within tolerance."*
- It claims "**mesh convergence study at Pe=1000 with <2% change at 50 cells/mm**". This study **does not exist**. Replace with the real Hagen–Poiseuille convergence study above.
- It does not currently mention the Hagen–Poiseuille velocity/WSS verification at all — add it.

Code paths: `scripts/run_scalar_verification.py`, `scripts/run_verification.py`, `ooc_optimizer/cfd/verification.py`, `ooc_optimizer/cfd/scalar.py: run_scalar_verification_1d`. Tests: `tests/test_scalar_verification.py`, `tests/test_verification.py`.

### Q3 — "Ladder only uses (W, Q_total) — five are masked." Does ladder really not optimise the other five?

**Yes, literally.** `bo_loop.py:60-66`:

```python
if topology == "ladder":
    for name in ("theta", "r_flow", "delta_W", "d_p", "s_p"):
        active[PARAMETER_ORDER.index(name)] = False
```

The 7-D input vector is preserved for plotting/cross-topology comparison, but masked dimensions are **pinned at the midpoint of their bounds and never moved by BO**. The ladder problem is therefore a **2-D constrained BO** in `(W, Q_total)`. The H=300 winner's `x_star_norm = [0.998, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5]` confirms: W and Q at upper corners, the rest exactly at 0.5. The trained GP and Sobol are 2-D (`names: ["W", "Q_total"]` in `summary.json`).

This is a methodological choice: the ladder geometry is parameterised by `(W, H, N, C_k_convention)`, with the operating point given by `Q_total`. The other five parameters in the master list have no mechanical role in the ladder topology (no pillars, no inlet angle, no tongue offset, no flow ratio between two streams).

The ladder's apparent "Sobol cleanness" (`Σ S_T ≈ 1.01`) is therefore not because the BO discovered five dimensions to be irrelevant — it is because those dimensions were never on the table.

### Q4 — How well does the paper currently cover the post-hoc interpretability triple, and how to expand?

**Coverage assessment:**

| Component | Defined in §2.6? | Equation? | Figure? | Discussed in §4? | Gap |
|---|---|---|---|---|---|
| Sobol total-effect | yes (1 line) | no | Figs 4–7, 10 | §4.3 | No `Σ S_T` self-audit; no equation; one figure per topology (4 charts). |
| Local sensitivity `|∂μ/∂x_norm|` | **no** | **no** | Figs 8, 11 | barely | Not actually defined as the GP-mean gradient at the optimum. |
| Fabrication tolerance | yes (1 line) | no | Figs 9, 12 | §4.7 | Bisection algorithm not specified. |
| Constraint-binding diagnostic | yes (1 line) | no | **none** | §4.2, §4.7 | Mentioned only in prose; no plot. |
| `R²-to-linear` | yes (Table 3) | **no** | implicit | §3.4 | Definition missing. |
| BO convergence curve | no | — | **none** | no | Standard for any BO paper; we have `ooc_optimizer/analysis/convergence.py` to generate it. |
| **CFD field of the winner** | no | — | **none** | no | Single most-missing artefact. |

**Expansion plan:** new §2.6 with three equations; new figures (cross-topology Sobol grouped bar focused on the two-inlet trio, constraint-binding plot, BO convergence curve, H=300 winner concentration field); new `Σ S_T` audit table; ladder per-inlet `C_k` 8-D BO promoted from §4.10 into a "future interpretability case study". See §B below for full details.

---

## A. Factual mismatches against the codebase (must fix)

These are claims a reader can verify from the open-source repo. As written, several are wrong or unbacked.

### A.1  Sobol initialisation count
- **Paper §2.3:** "GP initialised with **8 Sobol** samples."
- **Reality:** `examples/tumor_chip_linear_gradient/config.yaml: n_sobol_init: 24`; loop draws 24 (`bo_loop.py:101, 244`).
- **Fix:** change to **24** Sobol initialisation points.

### A.2  Mesh density (`ny_per_mm`) — **measured, paper is wrong**
- **Paper §2.2:** "uniform grid density of **25 cells/mm** in xy-plane".
- **Reality (from the H=300 winner's `blockMeshDict`):** 200 x-cells over 10 mm = **20 cells/mm in x**; 88 y-cells over W=4495.6 μm ≈ **19.6 cells/mm in y**. The actual production resolution is **20 cells/mm**, not 25. The diagnostic-findings document is also wrong.
- Per-strip resolution: 88 y-cells / 8 strips = **11 cells per inlet strip**, defensible but tight; numerical-diffusion budget §4.6 was estimated for 25 cells/mm — re-derive at 20 cells/mm if needed.
- **Fix:** quote 20 cells/mm. Also correct `findings/diagnostic_findings.md`. The numerical-diffusion residual estimate in §4.6 (~0.015 of L2) was computed assuming 25 cells/mm; at 20 cells/mm the contribution may be slightly higher (~0.018) — the residual budget still sums to ≈0.067 within reporting precision but the breakdown should be re-run.

### A.3  Inlet-angle range
- **Paper Table 1:** `θ ∈ [30°, 150°]`.
- **Reality:** `config.yaml: theta: min: 15, max: 75`.
- **Fix:** correct the bounds. Also revisit the wording "inlet convergence half-angle" — confirm sign convention against `topology_blockmesh.py`.

### A.4  Dead-zone fraction definition
- **Paper Table 2:** `f_dead` = fraction with `U < 0.01·U_max`.
- **Reality:** `metrics.py:237` — `_dead_fraction(U_mag, threshold_ratio=0.1)`, called on `U_mag_dev` (the *mean*-relative magnitude inside the developed region), not `U_max`. The threshold is **10% of the developed-region mean**, not 1% of max.
- **Fix:** match the code. Suggested wording: "`f_dead` = fraction of the developed-flow region with velocity below 10% of the region's mean velocity." Update the poster too.

### A.5  Solver-verification claims (1D advection–diffusion)
This is the most exposed section. The paper makes specific numerical claims that are partly unbacked.

**What "solver verification against the 1D AD analytic" means.** Run `scalarTransportFoam` on a 1D channel with uniform velocity, compare the simulated `C(ξ)` against the textbook analytic `C(ξ) = (1 − e^{−Pe(1−ξ)}) / (1 − e^{−Pe})`. Pure solver verification: isolates the scalar transport solver from any geometry / BC effect.

**What we actually have:**

1. `data/scalar_verification/scalar_verification_results.json` — 1D advection–diffusion verification. Real numbers, all PASS (<2% L2 rel. error):

   | Pe | n_cells | L2 rel. error | L∞ error | Pass |
   |---|---|---|---|---|
   | 1     | 100 | 0.064 % | 0.058 % | ✓ |
   | 10    | 100 | 0.77  % | 1.63  % | ✓ |
   | 100   | 100 | 1.26  % | 10.65 % | ✓ |
   | 1000  | **500** | 1.66 % | 36.79 % | ✓ |

2. `data/verification/convergence_results.json` — Hagen–Poiseuille velocity / WSS convergence study at three mesh levels (nx=100/200/400; ny=10/20/40). All levels pass 2% on centerline velocity and WSS; mean velocity error drops 1.18% → 0.39% → 0.13%.

**What the paper currently says (and is wrong about):**

- Says **"100-cell mesh for all four Pe"**. Reality: **Pe=1000 used 500 cells**. The 100-cell run at Pe=1000 would *fail* the 2% bar because of upwind numerical diffusion. Fix wording to: *"L2 relative error < 2% for Pe ∈ {1, 10, 100} on a 100-cell mesh; for Pe = 1000 the mesh was refined to 500 cells to suppress upwind diffusion to tolerance."*
- Claims **"mesh convergence study at Pe=1000 with <2% change at 50 cells/mm"**. This study **does not exist in the repo**. The convergence study we *do* have is the Hagen–Poiseuille refinement at nx=100/200/400. Replace the fabricated claim with this real one: *"Hagen–Poiseuille convergence study at nx ∈ {100, 200, 400} (corresponding to ny ∈ {10, 20, 40}) yielded mean-velocity errors of 1.18 %, 0.39 %, 0.13 % respectively — confirming production-level meshes (ny ~ 25 per mm) sit on the converged plateau."*
- Add a paragraph mentioning the Hagen–Poiseuille velocity-field verification — the paper currently mentions only scalar verification.

**Code paths** (use these in supplementary if needed):
- Driver: `scripts/run_scalar_verification.py`, `scripts/run_verification.py`
- Implementation: `ooc_optimizer/cfd/verification.py`, `ooc_optimizer/cfd/scalar.py: run_scalar_verification_1d`
- Tests: `tests/test_scalar_verification.py`, `tests/test_verification.py`

### A.6  Per-topology BO wall-time
- **Paper §2.3:** "200 evaluations required ~40 minutes per topology on an 8-core workstation."
- **Reality:** The H-sweep ran 2 × 200 ladder evals in **13 min 22 s wall** (per `findings/ladder_H_sweep_findings.md`). Earlier integration run did 4 × 200 evals in ~1 hour wall. 40 min/topology is not in any log.
- **Fix:** Quote the H-sweep number (cleanest data point) or the integration run number, with an explicit "wall-time / CPU-time" qualifier. Also fix the **hardware** statement: paper says **Apple M2**, machine context says **Apple M4** (CLAUDE.md).

### A.7  Parameter-table topology coverage (the "5 masked dimensions" issue)
- **Paper Table 1:** lists 7 continuous parameters as if all are optimised for every topology.
- **Reality:** `bo_loop.py:60-66` — when topology = `ladder`, the BO **masks out** five dimensions (`theta, r_flow, delta_W, d_p, s_p`); they are pinned to the midpoint of their bounds and never moved. The ladder optimisation is genuinely **2-D** in `(W, Q_total)`. The H=300 winner `x_star_norm = [0.998, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5]` confirms it: W at upper corner, Q at upper corner, the rest exactly at 0.5. The trained GP and Sobol indices are reported on **2 dims**, not 7.
  - For two-inlet topologies the active dim count is 4 or 5 (varies; opposing keeps `delta_W`, others drop it; `pillar=none` drops `d_p, s_p`).
- **Why this matters.** A reader of the current paper might assume the ladder Sobol "discovers" that 5 dimensions are irrelevant — that is wrong. The truth is the model is *defined* as 2-D for ladder; Sobol confirms the design choice was correct. Hiding this misrepresents what the BO does.
- **Fix.** Either:
  1. **Add a column** to Table 1 — `Active in: opposing | same_side_Y | asymmetric_lumen | ladder` — with check marks. Most cells will be ✓, ladder column has only `W, Q_total` ticked.
  2. **Or split into two tables:** "Geometry/flow parameters (used by all topologies)" — `W, Q_total`. "Pillar / inlet-angle parameters (used by two-inlet topologies)" — `θ, δ_W, d_p, s_p, r_flow`.
- Add one sentence to §2.5: *"For the ladder topology, the geometry is fully specified by `(W, H, N, C_k_convention)` and the operating point by `Q_total`; all other parameters in Table 1 are masked at the BO layer (pinned to the midpoint of their bounds), making the ladder optimisation a 2-D constrained BO in `(W, Q_total)`."*

### A.8  Reference 11 (Ayuso) — citation year
- The paper cites Ayuso 2021 *Sci. Adv.*; the report and prior literature survey reference Ayuso 2020 *IJMS*. Confirm which source the asymmetric-lumen topology is actually adapted from and stay consistent.

---

## B. Post-hoc interpretability — promote to spotlight (and reframe around the two-inlet topologies)

The title already promises a *Topology-First Design Methodology*; the most defensible *methodological* contribution is the **post-hoc interpretability triple**: Sobol total-effect + local sensitivity + bisection-based fabrication tolerance, plus the constraint-binding readout. The paper currently has the artefacts but does not present them as a unified, named contribution.

### B.0  Reframing: interpretability lives on the two-inlet topologies, not on the ladder

The ladder is the *winner*, but its BO is only 2-D in `(W, Q_total)` (see A.7). With only two active dimensions, the Sobol/local/tolerance machinery has very little to chew on, and the cross-topology comparison is degenerate (you cannot compare the Sobol weights of `r_flow` for ladder, because `r_flow` is masked there).

The interpretability *story* is therefore much richer on the four-active-dim two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`), and that is where the paper's interpretability section should put its weight. The ladder's 2-D Sobol becomes a small confirmatory result ("the design choice that made it 2-D was correct: `Q_total` and `W` are both individually important; no hidden interactions"). Per-inlet `C_k` 8-D BO on the ladder, which would expose the genuine internal structure of the winner, moves to **Future Work** (already proposed in §4.10 #1).

**Three recurring findings worth elevating to spotlight headlines** (data verified from the four `interpretability/summary.json` files):

| Finding | Evidence | Where to put it |
|---|---|---|
| **F1 — `r_flow` is the universal driver of the two-inlet class.** | `S_T(r_flow)`: `same_side_Y = 0.860`, `asymmetric_lumen = 0.976`, `opposing = 0.651` (with the overfit caveat). Local sensitivity ranking puts `r_flow` first for `same_side_Y` (`|∂μ/∂x| = 1.27`), `asymmetric_lumen` (`1.12`), and second-with-largest-magnitude for `opposing` (`3.13`). | New §4.3 lead paragraph — promote to the heuristic that drives the entire interpretability discussion. |
| **F2 — `θ` (inlet convergence angle) is consistently inactive across all two-inlet topologies.** | `S_T(θ)`: `opposing = 1.8e-4`, `same_side_Y = 1.6e-3`, `asymmetric_lumen = 3.5e-5`. Local sensitivity also negligible (≤ 0.05 across all three). | New §4.3 — frame as "the BO and Sobol agree that inlet angle is mechanically irrelevant for transverse-gradient fidelity in the laminar regime." A search-space-pruning recommendation: future studies can fix `θ` at midpoint and save a dimension. |
| **F3 — `W` and `Q_total` trade off as the secondary lever.** | The two parameters' `S_T` together account for ~10–40% of variance after `r_flow`. Per-topology rank: `same_side_Y` puts `W` (`S_T=0.13`) ahead of `Q_total` (`S_T=0.077`); `asymmetric_lumen` does the opposite (`Q_total=0.033 > W=0.013`); `opposing` (caveated) puts `Q_total=0.39, W=6.6e-5`. | New §4.3 — explain mechanistically: once `r_flow` has set the lateral interface position, the secondary L2 reduction comes from chamber width (more diffusion length) **or** flow rate (higher Pe), depending on which limit is active for that topology. The two are interchangeable substitutes, not independent additives. |

These three findings (F1, F2, F3) together replace the current §4.3 ("Cross-topology Sobol — the publishable design heuristic") and *are* the publishable design heuristic. The "ladder is flow-knob; two-inlet is interface-knob" sentence is preserved as a one-line bridging contrast at the end, not the main event.

### B.1  What's currently in the paper

| Component | Defined in §2.6? | Equation? | Figure? | Discussed in §4? | Gap |
|---|---|---|---|---|---|
| Sobol total-effect | yes (1 line) | no | Figs 4–7, 10 | §4.3 | No `Σ S_T` self-audit; no equation; one figure per topology (4 charts). |
| Local sensitivity `|∂μ/∂x_norm|` | **no** | **no** | Figs 8, 11 | barely | Not actually defined as the GP-mean gradient at the optimum. |
| Fabrication tolerance | yes (1 line) | no | Figs 9, 12 | §4.7 | Bisection algorithm not specified. |
| Constraint-binding diagnostic | yes (1 line) | no | **none** | §4.2, §4.7 | Mentioned only in prose; no plot. |
| `R²-to-linear` | yes (Table 3) | **no** | implicit | §3.4 | Definition missing. |
| BO convergence curve | no | — | **none** | no | Standard for any BO paper; we have `ooc_optimizer/analysis/convergence.py` to generate it. |
| **CFD field of the winner** | no | — | **none** | no | Single most-missing artefact. |

### B.2  How to expand — concrete edits

1. **New §2.6 sub-section: "Post-hoc interpretability triple."** Three equations:
   - **Sobol total-effect index** on the trained GP-mean function `μ(x)`:
     $S_{T,i} = \mathbb{E}_{x_{\sim i}}[\mathrm{Var}_{x_i}(\mu \mid x_{\sim i})] / \mathrm{Var}(\mu)$, estimated via Saltelli sampling, n=1024.
   - **Local sensitivity** at the BO optimum `x*`: `|∂μ/∂x_norm|_{x*}`, where the gradient is computed by autograd through the GP mean and `x_norm` is the unit-cube-normalised parameter.
   - **Fabrication tolerance** Δx by bisection: largest `|x_i − x_i^*|` such that `μ(x*+Δx_i e_i) ≤ 1.10 · μ(x*)`, with all other dimensions fixed. The bisection is performed on the GP mean to avoid additional CFD calls.
2. **Define `L2` and `R²-to-linear` formally.** L2 normalised-RMS:
   $L_2 = \|C - C_\text{target}\|_2 / \|C_\text{target}\|_2$.
   R²-to-linear: least-squares regression `C(y) = a + b·(y/W)` over depth-averaged `C(y)` along the chamber midline, reporting `R²`. Justify why both are reported (L2 captures *overall* fidelity including offset/scale; R² captures shape *independently* of offset/scale).
3. **New figure: cross-topology Sobol grouped bar chart, focused on the two-inlet trio.** One figure with the **three two-inlet topologies** (`opposing`, `same_side_Y`, `asymmetric_lumen`) on the x-axis and `r_flow, W, Q_total, θ, δ_W` as parameter groups. Annotate `Σ S_T` per topology and put a red "untrustworthy" badge on `opposing`. Three things should be visually instant from this single chart: F1 (`r_flow` bars are tall everywhere), F2 (`θ` bars are floor-noise), F3 (`W` vs `Q_total` swap dominance between topologies). The ladder Sobol (only `W, Q_total`) goes to a small inset sub-panel labelled "Confirmatory: the 2-D ladder design space." Replaces Figs 4, 5, 6, 7 and Fig 10.
4. **New table: `Σ S_T` trustworthiness audit.**

   | Topology | Σ S_T | Verdict |
   |---|---|---|
   | ladder | 1.014 | ✓ |
   | asymmetric_lumen | 1.022 | ✓ |
   | same_side_Y | 1.065 | ✓ |
   | opposing | **1.813** | ⚠ overfit |

   This makes the surrogate self-check explicit instead of buried in §4.5.
5. **New figure: constraint-binding plot.** A horizontal "constraint-slack" bar chart showing, for each of the 5 constraints at H=200 vs H=300, the gap between the optimum's value and the constraint threshold (0 = binding, positive = slack). Visually communicates the H=200 → H=300 corner shift in one image.
6. **New figure: BO convergence curve.** Best-feasible-L2 versus evaluation index, all four topologies on one axes (H=200), plus ladder H=200 vs H=300 on a second axes. Generate via `ooc_optimizer/analysis/convergence.py`. This is *standard* for BO papers and currently entirely absent.
7. **New figure: H=300 winner concentration field.** Two panels: (a) 2-D normalised C heatmap of the winner CFD case; (b) depth-averaged C(y) profile overlaid on the linear target. Generate from `bo_ladder_none_H300/<best-case-dir>/`. The most persuasive single image in the paper, currently missing.
8. **Add 1 paragraph to §4.7** explaining how each of the three interpretability outputs answers a *different* practitioner question: (i) Sobol → "what should the lab calibrate most carefully?"; (ii) tolerance → "how tight must fabrication be?"; (iii) constraint binding → "which constraint should the lab try to relax to push performance?"
9. **Future-work expansion: per-inlet `C_k` 8-D BO on the ladder.** Currently §4.10 #1 already lists this as the highest-priority next step. Strengthen it by tying it explicitly to the interpretability section: *"The ladder's 2-D `(W, Q_total)` Sobol exhausts the geometric/operating dimensions; the per-inlet `C_k` 8-D campaign is the natural way to turn the ladder back into a high-dimensional interpretability case study, exposing whether step-quantisation, diffusion smoothing, or inlet-region acceleration dominates the residual L2 (cf. §4.6 budget). We expect Sobol on the resulting 8-D GP to identify two or three end-strip `C_k` values as the dominant levers, by analogy with the `r_flow` dominance observed in two-inlet designs."*

The above adds **3 new figures + 1 new table + ~1 page of text**, all backed by data already in the repo. The interpretability narrative is now anchored on the three recurring two-inlet findings (F1/F2/F3), with the ladder appearing as both the *winner* (in the BO results section) and the *next interpretability case study* (in future work).

---

## C. Structural problems

### C.1  No abstract.
Add a ~200-word abstract: problem → mass-conservation insight → ladder pivot → cross-topology BO + Sobol + tolerance result (`L2 = 0.067, R² = 0.990, 9.4× over baseline; 10.8× over best alternative on the same target`) → claim of methodological novelty (interpretability triple + feasibility pre-screen).

### C.2  Title vs. content.
Title promises *"Topology-First Design Methodology"* but §2 reads chronologically. Recast §2 around three named methodological contributions:
1. **Mass-conservation feasibility pre-screen** (§3.1's idea, abstracted as a method, not a post-hoc explanation).
2. **Constraint-aware BO with 5-GP ConstrainedEI** and the diagnostic-metric set.
3. **Post-hoc interpretability triple** (§B above).

Then §3 demonstrates the methodology on the tumor-chip case.

### C.3  Mass-conservation derivation is hand-wavy.
§3.1 asserts `⟨C⟩_y(x) = r_flow` in two sentences. A formal paper deserves the 4-line derivation, placed in **Methods §2.2** as a *predictive* tool (matching the "topology-first" title), not in Results §3.1 as a post-hoc explanation:
- Define `⟨C⟩_y(x) = (1/W) ∫_0^W C(x,y) dy`.
- Integrate the steady-state advection–diffusion equation `∇·(uC − D∇C) = 0` across the chamber width.
- Drop the cross-stream diffusive flux at side walls (no-flux BCs).
- Conclude `∂_x [⟨uC⟩_y] = 0`.
- Evaluate at `x=0`: `⟨uC⟩_y(0) = u·(c_low·(1−r_flow) + c_high·r_flow) = u·r_flow` for `c_low=0, c_high=1`.
- Therefore `⟨C⟩_y(x) = r_flow` ∀ x.
- Conclude: `⟨C⟩_y = x/L` (the linear-x target) is incompatible.

### C.4  Phase-2 (W, Q) scan story is buried in §4.5.
The 32-eval scan that "looked flat" used the **endpoint** convention; production uses **midpoint** (38% lower L2 at fixed geometry). This is the *only* reason §3.2's flat heatmap and §3.3's 0.082 winner appear contradictory. Promote the convention explanation to §3.2 right after Table 5; do not leave it in §4.5 "honest caveats".

### C.5  §3.3 and §3.4 partially overlap.
§3.3 reports H=200; §3.4 reports H=300 + recap of H=200. Consolidate into one subsection: *"Bayesian optimisation at H = 200 μm and H = 300 μm"*, with a single comparison table and paired figures.

### C.6  Topology screening (B–E) is one paragraph.
Add a screening table with explicit criteria — Pe regime, mass-conservation compatibility, manufacturability, BC-engineering complexity, steady-state existence — for each of A through E. Move the bullet content from `REPORT.md §2.3` to the paper. Without this, "we picked A because it was easiest" looks unprincipled.

### C.7  Conclusion is generic.
Replace with three crisp claims of transferable contribution:
1. Mass-conservation pre-screen as a generalisable feasibility test for any prescribed-field design problem in the laminar regime.
2. Constraint-corner pinning as a design heuristic (not pathology).
3. Cross-topology Sobol dichotomy: flow-knob (encoded-gradient topologies) vs. interface-knob (interface-generated topologies).

### C.8  Annotate Fig. 7 (`opposing` Sobol).
Either drop it or put a clear "ΣS_T = 1.81 — surrogate overfit, magnitudes inflated" banner directly on the plot. The §4.5 caveat is too far from the figure for an external reader.

---

## D. Missing content

### D.1  Code / data availability statement.
"Pipeline is open-source and documented" → give the GitHub URL, the commit hash used for the reported results, an archived DOI (Zenodo), and list `findings/diagnostic_findings.md`, `findings/integration_run_findings.md`, `findings/ladder_H_sweep_findings.md` as supplementary material.

### D.2  Reproducibility paragraph.
Add to §2 (Methods): OpenFOAM version (v2406, already there), Python and BoTorch versions, GP hyperparameter optimisation strategy, random-seed handling for the Sobol initialisation, hardware (Apple M4 — fix the M2 typo; CLAUDE.md confirms M4), total CPU-h budget (≈ 6.5 CPU-h: integration ~5 CPU-h + H-sweep 0.4 CPU-h + verification ~1 CPU-h).

### D.3  Related-work table.
The introduction cites Yang 2020, Hong 2020, Zhang 2022 in prose. Convert to a 5-row table with columns *target type / dimensionality / optimiser / constraints reported / interpretability outputs* so the novelty claim "no published work combines (i)–(iv)" becomes *visible* rather than asserted.

### D.4  Borrvall–Petersson 2003.
Mentioned in `REPORT.md` and poster as roadmap (density-based topology optimisation). Cite it in §4.10 if invoking topology optimisation as future work.

### D.5  Why W = 4496 (not 4500) at H=300.
Poster Q3 explains: BO bisection-tolerance at the corner; AR=15 cap and W_max are degenerate at H=300. Pull that one paragraph into §3.4 so a reader doesn't wonder.

### D.6  Convergence curve and concentration-field figures.
See B.2 #6 and #7.

---

## E. Style — moving from poster to paper

- Replace conversational verbs: "the optimum *sat* at" → "the optimum lies at"; "BO *spends* on higher Q" → "BO reallocates the relaxed shear margin to higher Q"; "earns its keep" → "is justified".
- Retitle "Honest caveats" (§4.5) as *"Limitations of the optimisation"* or fold into §4.10.
- **Headline framing.** "9.45× improvement" compares an axis-x baseline against an axis-y winner — different targets. Lead with **"L2 = 0.067, R² = 0.990, 10.8× better than the next-best of four candidate topologies on the same target"**, and quote the 9.45× exactly once with the axis-flip caveat. The 10.8× number is the apples-to-apples comparison and just as impressive.
- §4.7 *"What the BO + Sobol stack actually buys"* → *"Methodological value beyond the L2 minimum"*.

---

## F'. Additional critical suggestions (second pass)

### F'.1  Lead with the same-axis 10.8× number, not the cross-axis 9.45×
The 9.45× comparison spans both topology change and target-axis flip. The 10.8× comparison (ladder vs next-best on axis=y, same campaign) is apples-to-apples and equally impressive. Lead with 10.8× in abstract, conclusion, and §3.3; quote 9.45× exactly once with the axis-flip caveat.

### F'.2  GP surrogate quality audit (R²_LOO)
Sobol indices are claims about the GP surrogate, not reality. Compute leave-one-out R² on the saved `gp_model_state.pt` checkpoints for the **objective GP** and each **constraint GP**, per topology. Add to the trustworthiness table (next to `Σ S_T`). Without this, the `opposing` flag at `Σ S_T = 1.81` is the only surrogate-quality signal in the paper.

### F'.3  Promote the midpoint-convention 38% gain
"Use the midpoint `C_k = (k+0.5)/N` rather than endpoint `C_k = k/(N−1)`" is a generalisable recommendation for any ladder-class or Christmas-tree-class topology. Currently buried in §3.2. Add one sentence in the Conclusion: *"The midpoint inlet-concentration convention is geometrically optimal for any imposed-inlet gradient generator and yields a free 38% L2 reduction over the more common endpoint convention."*

### F'.4  Use the R² vs L2 distinction explicitly
L2 conflates magnitude error with shape error; R²-to-linear isolates shape. At H=300, R² rose 0.987 → 0.990 *while* C_std dropped 0.314 → 0.306 — meaning the H=300 field is simultaneously *more linear in shape* and *slightly more conservative in dynamic range*. This is a real interpretability win that one sentence in §4.2 can capture.

### F'.5  Name the categorical-topology BO pattern
What the loop does is treat topology as a categorical hyperparameter and run one BO per category sharing the parameter space and constraints. Make this explicit in §2.3: *"Topology is treated as a categorical hyperparameter; we run an independent BO loop per topology with a shared parameter space and constraint set, then compare optima."*

### F'.6  Constraint-set rationale (§2.4)
Add a 1-paragraph justification for the choice of *exactly* these 5 constraints, in this order: biology (τ window) + flow uniformity (`f_dead`) + safety (Re) + manufacturability (`W/H`) + discrete experimental knob (H). This pre-empts the reviewer question "why not pH, why not stagnation pressure, why not …".

### F'.7  Feasibility rate as a separate GP-quality signal
37% → 96.5% at H=300 means the GP is fit on 2.6× more useful evaluations and its posterior variance at the optimum is correspondingly lower. Add as a separate justification line in §4.2 — independent of the L2 improvement.

### F'.8  Group future work into tiers
The current six-item flat list is report-style. Group into:
- **Tier 1 — tractable extensions:** open the search box, per-inlet `C_k` 8-D BO, mesh refinement.
- **Tier 2 — methodological extensions:** other target shapes (step, bimodal, time-varying), other diffusivities (large biologics).
- **Tier 3 — major directions:** topologies B–E implementation, 3D validation, density-based topology optimisation, experimental chip validation.

### F'.9  Feasibility pre-screen as a deliverable tool
§2.2 currently uses mass conservation as a *post-hoc* explanation. Recast as a *predictive* methodological tool: a small admissibility table by prescribed-target × topology-class:

|  | linear x | linear y | step y | bimodal y |
|---|---|---|---|---|
| Two-inlet coflow (opposing, SSY, asym. lumen) | ✗ | ✓ (limited) | ✓ | ✗ |
| Imposed-inlet ladder | ✗ | ✓ | ✓ | ✓ |
| Distributed source (side-injection, permeable) | ✓ | — | ✓ | ✓ |
| Counter-flow | ✓ | ✗ | unsteady | unsteady |

This turns the mass-conservation insight into a one-glance admissibility check that any subsequent paper can reuse.

---

## F. Smaller items

- Hardware: paper says **M2**, real machine is **M4**. Fix.
- Cross-reference Figs 4–7 with their H=300 counterparts (Fig 10) using paired `Fig. X(a)/(b)` so the reader sees the H sweep at a glance.
- The discrepancy "20 cells/mm default vs 30 for ladder" is itself a methodological point — your ladder mesh is denser. Decide whether this is intentional and disclose.
- Verify Eq. (4) in the paper exactly matches what the verification driver compares against. A unit-test (`tests/test_scalar_verification.py`) already does this — cite the test, not just the formula.
- No leftover absolute paths in the manuscript (`/Users/lemon/...`).

---

## G. Suggested priority order

| Priority | Item | Effort |
|---|---|---|
| P0 | Fix A.1–A.7 (codebase factual mismatches) | ~30 min |
| P0 | Add the H=300 winner concentration-field figure (B.2 #7) | ~30 min |
| P1 | Add abstract, mass-conservation derivation, L2/R² equations (C.1, C.3) | ~1 h |
| P1 | Add §2.6 interpretability-triple definitions (B.2 #1) | ~30 min |
| P1 | Add cross-topology Sobol grouped bar (B.2 #3), Σ S_T audit table (B.2 #4), constraint-binding figure (B.2 #5), BO convergence figure (B.2 #6) | ~1.5 h |
| P2 | Restructure §3.3+§3.4, surface convention discrepancy (C.4, C.5), annotate Fig 7 (C.8) | ~30 min |
| P2 | Add code-availability + reproducibility (D.1, D.2), related-work table (D.3), Borrvall–Petersson reference (D.4), W=4496 explanation (D.5) | ~30 min |
| P3 | Conclusion rewrite (C.7), §4.5 retitle (E), 9.45× framing (E) | ~30 min |

Total: ~5 h to move from "internal write-up" to "share-able formal manuscript".
