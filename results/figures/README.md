# Figures — Where They Live

This folder is intentionally empty of binary assets. Every figure used in the manuscript and poster lives in one of two upstream locations, and is referenced from there to avoid duplication. This README maps each figure name to its on-disk path so the poster-prep workflow can locate them quickly.

---

## Manuscript figures (LaTeX `bayesian_src/main.tex`)

All paths are relative to the repository root.

| LaTeX label | Filename | Path | Description |
|---|---|---|---|
| `fig:topology_candidates` | `A_ladder_N8 (1).png` | `bayesian_src/A_ladder_N8 (1).png` | Schematic of stacked ladder topology |
| | `B_christmas_tree.png` | `bayesian_src/B_christmas_tree.png` | Christmas-tree mixer schematic |
| | `C_side_injection_K8.png` | `bayesian_src/C_side_injection_K8.png` | Distributed side-injection schematic |
| | `D_permeable_wall.png` | `bayesian_src/D_permeable_wall.png` | Permeable-wall schematic |
| | `E_counter_flow.png` | `bayesian_src/E_counter_flow.png` | Counter-flow schematic |
| `fig:phase2_scan` | `phase2_W_Q_scan.png` | `bayesian_src/phase2_W_Q_scan.png` | Sobol-scan response surface, endpoint convention |
| `fig:bo_convergence` | `fig_d_bo_convergence.png` | `bayesian_src/fig_d_bo_convergence.png` | BO convergence curves (4-topology + H-sweep) |
| `fig:concentration_field` | `fig_e_concentration_field.png` | `bayesian_src/fig_e_concentration_field.png` | H = 300 ladder winner CFD field + linearity check |
| `fig:constraint_binding` | `fig_b_constraint_binding.png` | `bayesian_src/fig_b_constraint_binding.png` | Constraint-binding diagnostic across topologies |
| `fig:sobol` | `fig_c_cross_topology_sobol.png` | `bayesian_src/fig_c_cross_topology_sobol.png` | Cross-topology Sobol with F1/F2/F3 annotations + ladder confirmatory |
| `fig:pillar_swap` | `fig_g_pillar_regime_swap.png` | `bayesian_src/fig_g_pillar_regime_swap.png` | Sobol regime swap (pillars=none vs 1×4) |
| `fig:pillar_field` | `fig_h_pillar_field.png` | `bayesian_src/fig_h_pillar_field.png` | Pillar 1×4 winner CFD field |

The LaTeX manuscript is at `bayesian_src/main.tex` and is the most up-to-date source (verified 2026-04-30 to be newer than the now-superseded `bayesian.zip` archive).

---

## Poster figures

The poster's figures are organised by panel under `poster/figures/`:

| Panel folder | Contents |
|---|---|
| `01_topology_candidates/` | A–E topology schematics (same as manuscript above, possibly higher resolution) |
| `02_cross_topology_summary/` | `cross_topology_summary.png` — best-feasible-L2 bar chart across the 4 topologies |
| `03_sobol_per_topology/` | `sobol_<topology>_H<H>.png` — per-topology Sobol bars |
| `04_local_sensitivity/` | `local_sensitivity_ladder_H<H>.png` — local sensitivity at the ladder optimum |
| `05_tolerance/` | `tolerance_ladder_H<H>.png` — fabrication tolerance intervals |
| `06_phase2_scan/` | `phase2_W_Q_scan.png` — same as manuscript |
| `paper_v2/` | The current paper-v2 figure set (mirrors `bayesian_src/` figures, plus `table_f_sigma_st_audit.md` for the Σ S_T audit table) |

Use the panel-folder layout when laying out the poster; use `bayesian_src/` figures when checking the manuscript build.

---

## Figure-generation scripts

`scripts/paper_figures/` contains the Python scripts that regenerate the manuscript figures:

| Script | Produces |
|---|---|
| `fig_b_constraint_binding.py` | `fig_b_constraint_binding.png` |
| `fig_c_cross_topology_sobol.py` | `fig_c_cross_topology_sobol.png` |
| `fig_d_bo_convergence.py` | `fig_d_bo_convergence.png` |
| `fig_e_concentration_field.py` | `fig_e_concentration_field.png` |
| `fig_g_pillar_regime_swap.py` | `fig_g_pillar_regime_swap.png` |
| `fig_h_pillar_field.py` | `fig_h_pillar_field.png` |

These reproduce the figures from the BO state checkpoints and JSONL eval logs in `examples/tumor_chip_linear_gradient/data/results/`. Re-running them after any data change is the canonical way to refresh the manuscript figures.

---

## Why no figures are duplicated into `results/figures/`

Two reasons:

1. **Single source of truth.** Duplicating PNGs would create three places where a figure could go stale (manuscript, poster, results). Keeping `bayesian_src/` and `poster/figures/` as the canonical locations preserves the existing build dependency.
2. **Disk-space discipline.** The repo root is on a 13 GB-free APFS volume; duplicating ~ 20 PNGs at ~ 100 KB each is harmless individually, but the rule "don't duplicate binary assets" generalises better.

If a poster-prep workflow needs a self-contained figures bundle, run:

```bash
mkdir -p results/figures/local
cp bayesian_src/*.png results/figures/local/
cp -r poster/figures/* results/figures/local/
```

…and let `results/figures/local/` be `.gitignore`d.
