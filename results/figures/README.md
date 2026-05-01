# Figures — Local Bundle for Presentation Prep

This folder contains every figure used in the manuscript and poster, copied locally so the presentation-prep workflow has a single self-contained location to draw from. The canonical sources are still `bayesian_src/` (manuscript) and `poster/figures/` (poster panels) — the copies here are for convenience and may go stale if upstream regenerates a figure.

---

## Folder layout

```
figures/
├── manuscript/              # All PNGs from bayesian_src/ (LaTeX manuscript figures)
└── poster_panels/           # Poster figures organised by panel
    ├── 01_topology_candidates/
    ├── 02_cross_topology_summary/
    ├── 03_sobol_per_topology/
    ├── 04_local_sensitivity/
    ├── 05_tolerance/
    ├── 06_phase2_scan/
    └── paper_v2/            # Paper-v2 figure set + Σ S_T audit table
```

---

## `manuscript/` — figures referenced by `bayesian_src/main.tex`

| LaTeX label | Filename | Description |
|---|---|---|
| `fig:topology_candidates` | `A_ladder_N8 (1).png` | Stacked ladder topology schematic |
| | `B_christmas_tree.png` | Christmas-tree mixer schematic |
| | `C_side_injection_K8.png` | Distributed side-injection schematic |
| | `D_permeable_wall.png` | Permeable-wall schematic |
| | `E_counter_flow.png` | Counter-flow schematic |
| `fig:phase2_scan` | `phase2_W_Q_scan.png` | Sobol-scan response surface, endpoint convention |
| `fig:bo_convergence` | `fig_d_bo_convergence.png` | BO convergence curves (4-topology + H-sweep) |
| `fig:concentration_field` | `fig_e_concentration_field.png` | H = 300 ladder winner CFD field + linearity check |
| `fig:constraint_binding` | `fig_b_constraint_binding.png` | Constraint-binding diagnostic across topologies |
| `fig:sobol` | `fig_c_cross_topology_sobol.png` | Cross-topology Sobol with F1/F2/F3 + ladder confirmatory |
| `fig:pillar_swap` | `fig_g_pillar_regime_swap.png` | Sobol regime swap (pillars=none vs 1×4) |
| `fig:pillar_field` | `fig_h_pillar_field.png` | Pillar 1×4 winner CFD field |
| (per-topology Sobol) | `sobol_<topology>_H<H>.png` | Per-topology Sobol bar charts |
| (per-H sensitivity) | `local_sensitivity_ladder_H<H>.png` | Local sensitivity at the ladder optimum |
| (per-H tolerance) | `tolerance_ladder_H<H>.png` | Fabrication-tolerance intervals |
| (overview) | `cross_topology_summary.png` | Cross-topology best-feasible-L2 summary |

---

## `poster_panels/` — figures organised by poster panel

The arrangement matches `poster/poster_draft.md`'s panel structure:

| Panel folder | Files | Used in |
|---|---|---|
| `01_topology_candidates/` | A–E topology schematics | Panel 2 — Topology Screening |
| `02_cross_topology_summary/` | `cross_topology_summary.png` | Panel 4 — Results: Cross-Topology BO |
| `03_sobol_per_topology/` | `sobol_<topology>_H<H>.png` (5 files) | Panel 4 — Cross-Topology Sobol |
| `04_local_sensitivity/` | `local_sensitivity_ladder_H<H>.png` (2 files) | Panel 5 — H-Sweep |
| `05_tolerance/` | `tolerance_ladder_H<H>.png` (2 files) | Panel 5 — H-Sweep |
| `06_phase2_scan/` | `phase2_W_Q_scan.png` | Panel 5 — H-Sweep |
| `paper_v2/` | `fig_b/c/d/e/g/h_*.png` + `table_f_sigma_st_audit.md` | Manuscript-quality versions of the same figures |

---

## How to refresh after upstream changes

If a figure is regenerated in `bayesian_src/` or `poster/figures/`, refresh this bundle with:

```bash
cd /Users/lemon/Desktop/ooc_loop
cp bayesian_src/*.png results/figures/manuscript/
cp -r poster/figures/01_topology_candidates poster/figures/02_cross_topology_summary \
      poster/figures/03_sobol_per_topology poster/figures/04_local_sensitivity \
      poster/figures/05_tolerance poster/figures/06_phase2_scan poster/figures/paper_v2 \
      results/figures/poster_panels/
```

The figure-generation scripts (canonical source for the `paper_v2/` figures) live in `scripts/paper_figures/` and read directly from `examples/tumor_chip_linear_gradient/data/results/`.
