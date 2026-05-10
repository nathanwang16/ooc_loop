# Tumor-on-Chip Inverse Design — Project Results

**Repository:** `ooc_loop`
**Reporting cutoff:** 2026-04-30 (after H-sweep and pillar-ablation campaigns).
**Author / contact:** Nathan Wang.  
**Status of this folder:** authoritative, first-principles narrative of *what was implemented and what it found*. All deeper documents are reachable from this README.

---

## 0. How to navigate this folder

This folder is the single source of truth for the project's principles, methods, and results. It is intentionally redundant with the LaTeX manuscript (`bayesian_src/main.tex`), the poster draft (`poster/poster_draft.md`), and the per-phase findings docs — those are written for specific external audiences; this folder is written so a reader who has never seen the project can rebuild the whole story from first principles.


| File                              | Audience         | Read first if you want…                                                                       |
| --------------------------------- | ---------------- | --------------------------------------------------------------------------------------------- |
| `README.md` (this)                | anyone           | a 5-minute headline picture                                                                   |
| `01_problem_and_principles.md`    | anyone           | why a linear gradient, why topology-first, what mass conservation forbids                     |
| `02_methodology.md`               | technical reader | the CFD + BO + interpretability pipeline, with formulas                                       |
| `03_topology_screening.md`        | designer         | the eight topologies and the admissibility logic                                              |
| `04_optimization_results.md`      | reviewer         | the four-topology campaign + H-sweep + pillar ablation, with winner geometries                |
| `05_interpretability_findings.md` | reviewer         | F1/F2/F3, fabrication-tolerance intervals, constraint-binding diagnostic                      |
| `06_translation_and_caveats.md`   | experimentalist  | the bench-ready chip spec + honest limitations                                                |
| `07_future_work.md`               | follow-up team   | tiered next steps                                                                             |
| `primary_sources/`                | auditor          | unchanged copies of the diagnostic / integration / H-sweep findings + the paper revision plan |
| `figures/README.md`               | reader           | which figure lives where (most live in `bayesian_src/` and `poster/figures/paper_v2/`)        |


If you have *one minute*: read §1 below. If you have *five*: read this whole file. If you have *one hour*: read this file plus `02_methodology.md` and `04_optimization_results.md`.

---

## 1. One-minute summary

We built a constrained-Bayesian-optimization (BO) pipeline that designs a microfluidic tumor-on-chip chamber producing a prescribed linear drug-concentration gradient across 3D-cultured cells. The campaign asked one question — *can BO find a chamber geometry whose CFD-simulated concentration field matches a linear ramp?* — and surfaced a surprising answer: **topology selection earns ~6× of the L2 reduction, before BO touches any parameter.** A 600-evaluation BO sweep across three two-inlet coflow topologies plateaued near a uniform-field floor (L2 ≈ 0.585) because mass conservation *forbids* a streamwise linear gradient in laminar two-inlet coflow. Pivoting to a Whitesides-style stacked-ladder topology aligned with the natural transverse transport, then tuning chamber height, lifted the design from infeasible (L2 ≈ 0.63) to publishable (L2 = 0.0671, R² = 0.990 at H = 300 μm) — a **9.4× reduction over the original target and a 10.8× reduction over the next-best topology on the same target**. A subsequent pillar-ablation campaign exposed a regime swap: introducing a single 1×4 pillar row drops the dominant Sobol parameter from `Q_total` to `W`, with another ~30% L2 drop available beyond the bare-ladder optimum.

**Headline numbers**


| Quantity                                                     | Value                    | Where      |
| ------------------------------------------------------------ | ------------------------ | ---------- |
| Best feasible L2 (H=300 ladder)                              | **0.0671**               | §4.2       |
| R²-to-linear at H=300                                        | **0.990**                | §4.2       |
| Improvement vs. axis=x baseline (cross-target)               | 9.4×                     | §1.3 below |
| Improvement vs. next-best topology, axis=y, same constraints | **10.8×**                | §4.1       |
| Best L2 with 1×4 pillars (constraint-relaxed)                | **0.0568**               | §4.3       |
| Feasibility at H=300                                         | 96.5% (vs. 37% at H=200) | §4.2       |
| Total CFD evaluations across the campaign                    | ~1300                    | —          |


---

## 2. Six general principles the project earned

These are the methodological take-aways. Each one is supported by data in the deeper documents and each one is transferable to other prescribed-field laminar-microfluidic problems.

1. **Mass-conservation pre-screen before any BO.** For any prescribed concentration field in laminar coflow with no through-wall scalar source, integrate the steady advection–diffusion equation across the chamber width and check whether the depth-averaged target is compatible with `⟨C⟩_y(x) = r_flow ∀ x`. If not, no design in that topology class can reach the target — *do not run BO*. (`01_problem_and_principles.md`, §3.)
2. **Topology selection dominates parameter optimization.** In our L2 stack, the topology pivot from two-inlet coflow to ladder buys ≈6× ; subsequent BO/constraint-relaxation layers add 1.2–1.3× each. Treat topology as the highest-priority design choice, not a fixed scaffolding to be tuned around.
3. **Constraint-corner pinning is informative, not pathological.** A monotone response surface always pins at the corner of the feasible box. Reading off *which* constraints bind (and which have slack) tells the experimentalist exactly which lab-side capability would most cheaply move the optimum. Our H = 200 → H = 300 transition is a textbook example: lifting a single parameter (chamber height) shifts the binding pair from (`AR`, `τ`) to (`AR`, `Q_max`), drops L2 by 17.9%, and triples feasibility — *without changing the optimizer*.
4. **The interpretability triple makes the optimum useful.** Sobol total-effect indices + local-sensitivity gradients + bisection-based fabrication-tolerance intervals + constraint-binding diagnostics turn a single L2 number into actionable lab guidance: *what to calibrate carefully, how tight fabrication has to be, which constraint to relax to push performance.*
5. **A two-inlet–vs–imposed-inlet dichotomy applies cross-topology.** Two-inlet coflow topologies are dominated by `r_flow` (the interface-position knob); imposed-inlet topologies (ladder) are dominated by `Q_total` (the residence-time knob). This is not a quirk of our parameter set — it follows from whether the gradient is *generated* by the chamber or *encoded* in the boundary condition.
6. **Mask, then audit by ablation.** Our ladder optimisation masked five of seven parameters under the assumption "only `W` and `Q_total` matter for the bare ladder." The pillar-ablation campaign empirically falsified that assumption *for the structured-medium case*: with a 1×4 pillar row, `S_T(W)` jumps to 0.86 and `S_T(Q_total)` collapses to ~0. **Any masked parameter that could plausibly couple to the response should be unmasked in a follow-up ablation, not just argued away.**

---

## 3. The story arc, in five movements

### 1.1 Diagnostic — the L2 floor is geometric

A 600-evaluation BO campaign over three two-inlet topologies (`opposing`, `same_side_Y`, `asymmetric_lumen`) against an axis-`x` linear gradient plateaued at **L2 = 0.6343** for the `opposing` winner — only 8% above the analytical uniform-field floor of 0.585. The mass-conservation argument (`01_problem_and_principles.md` §3) showed that no setting of the seven design parameters can break this floor because `⟨C⟩_y(x) = r_flow` is conserved along the chamber.

### 1.2 Topology pivot — five candidates, one implementation

Five replacement candidates were proposed and ranked by mass-conservation compatibility, manufacturability, and BC-engineering complexity. The Whitesides-style **stacked ladder** (axis-`y` target, N = 8 imposed-concentration inlet strips, midpoint convention `C_k = (k+0.5)/N`) was implemented end-to-end; the other four (`christmas_tree`, `side_injection`, `permeable_wall`, `counter_flow`) were screened in or out from first principles (`03_topology_screening.md`). A free 38% L2 reduction came from the midpoint vs. endpoint convention discovery alone, with no optimization needed.

### 1.3 Cross-topology BO at H = 200 — ladder wins by 10.8×

Under the new five-constraint feasibility set (`tau ∈ [0.1, 2.0]`, `f_dead ≤ 0.08`, `Re ≤ 100`, `aspect_ratio ≤ 15`, plus convergence/mesh validity), 200 evaluations per topology produced:


| Topology           | Best feasible L2 | Active dims    |
| ------------------ | ---------------- | -------------- |
| `**ladder`**       | **0.0818**       | 2 (W, Q_total) |
| `opposing`         | 0.8822           | 5              |
| `same_side_Y`      | 0.9937           | 4              |
| `asymmetric_lumen` | 1.0875           | 4              |


The ladder beats the next-best topology by **10.8×** on the same target. The 9.4× number that appears elsewhere compares the H = 300 ladder winner (axis-`y`) against the original axis-`x` BO winner — it spans both a topology change *and* a target-axis flip, so the apples-to-apples 10.8× is the headline. Detail in `04_optimization_results.md` §1.

### 1.4 H-sweep — corner shift unlocks 17.9% more

Opening the chamber height from 200 to 300 μm relaxes the `aspect_ratio_max = 15` cap from `W ≤ 3000` to `W ≤ 4500` μm. The H = 300 BO converged to a winner at `W = 4496 μm`, `Q_total = 200 μL/min` (cap), `L2 = 0.0671`, `R² = 0.990` — matching the dimensional-analysis pre-test prediction (`τ ∝ Q/(WH²)` → ~2.25× shear headroom that BO immediately spends on higher Q) within 2%. **Feasibility leapt from 37% to 96.5%** because the wider H pushes the τ-feasibility region to encompass most of the search box. The constraint-binding pair shifted from (`AR`, `τ`) at H = 200 to (`AR`, `Q_max`) at H = 300, with `τ_mean` releasing to 1.48 Pa (well below the 2.0 cap).

### 1.5 Pillar ablation — the regime swap

The bare-ladder optimisation masked five of seven design parameters (assuming pillars, inlet angle, flow ratio, and tongue offset are inactive for the ladder geometry). To audit that masking decision, we re-ran a 100-evaluation BO with `pillar_config = 1×4` (one row of four cylindrical pillars) at H = 200 μm. The active design space becomes 4-D in (W, d_p, s_p, Q_total). Two findings:

1. The best L2 falls to **0.0568** (a ~30% drop versus the bare-ladder H = 200 winner of 0.0818), with `R²_lin = 0.992`. Under the production constraint set (`τ ≤ 2.0` Pa) the best fully-feasible pillar design comes in at L2 ≈ 0.0588 — the gain is real but not yet fully captured under the production constraints.
2. The Sobol indices on the new 4-D surrogate **invert** the dominant control variable: `S_T(W)` jumps from 0.143 to **0.856**, while `S_T(Q_total)` collapses from 0.871 to **0.001**. `s_p` emerges as a meaningful secondary lever (`S_T = 0.139`); `d_p` stays minor (`S_T = 0.009`).

Mechanistic interpretation: with no pillars the chip is a parallel-plate channel and the gradient is set by residence time `L/U` → `Q_total` dominates; with pillars the chip becomes a structured medium and the gradient depends on `W/s_p` → `W` dominates. Detail in `04_optimization_results.md` §3.

---

## 4. Project artefacts at a glance

**Latest LaTeX manuscript:** `bayesian_src/main.tex` (verified 2026-04-30 to be the newest revision; the older `bayesian.zip` archive was deleted in the 2026-05-03 cleanup).

**Latest poster materials:** `poster/poster_draft.md`, `poster/figures/paper_v2/`, `poster/PAPER_REVISION_PLAN.md`. The per-phase findings docs that previously lived under `poster/findings/` are now consolidated in `primary_sources/` (single source of truth, see below).

**Per-phase analytical writeups (preserved verbatim):** `results/primary_sources/diagnostic_findings.md`, `results/primary_sources/integration_run_findings.md`, `results/primary_sources/ladder_H_sweep_findings.md`. These are the in-the-trenches analyses written immediately after each campaign — the highest-fidelity record of what we knew at each step.

**Code:** the BO + CFD + interpretability pipeline lives in `ooc_optimizer/` (Python package); reproducible scripts in `scripts/` and `examples/tumor_chip_linear_gradient/`. Figure-generation scripts for the manuscript are in `scripts/paper_figures/`.

**Hardware reproducibility:** Apple M4, macOS 26.3. OpenFOAM v2406. BoTorch (current). Python under conda env `ooc`. Total CPU-h budget across the reported campaigns: ~6.5 CPU-h.

**External submission target (planned):** the JOSS stub at `paper/joss/paper.md` is unrelated to the main manuscript — it is a software-paper draft for the BO pipeline as a tool. Out of scope for the poster presentation.