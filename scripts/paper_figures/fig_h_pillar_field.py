"""Figure (h) — Concentration field of the ladder × 1x4 pillar winner.

Two panels:
  (a) 2-D concentration field over the chamber, with the four pillar locations
      overlaid as outlines.
  (b) Depth-averaged C(y) profile vs. linear target, plus three downstream
      stations to show the geometric-mixer regime where streamline divergence
      around pillars re-mixes adjacent strips.

Source: examples/tumor_chip_linear_gradient/data/cases_ladder_pillar_ablation/
        run_ladder_1x4_H200_1777310180753_53206_2eb551/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ooc_optimizer.cfd.foam_parser import (  # noqa: E402
    find_latest_time,
    read_cell_centres,
    read_scalar_field,
)

CASE = REPO / (
    "examples/tumor_chip_linear_gradient/data/cases_ladder_pillar_ablation/"
    "run_ladder_1x4_H200_1777310180753_53206_2eb551"
)
OUT = REPO / "poster/figures/paper_v2/fig_h_pillar_field.png"

# Winner parameters
W_UM = 2121.3
H_UM = 200.0
L_MM = 10.0
N_STRIPS = 8
D_P_UM = 184.5
S_P_UM = 426.6
PILLAR_CONFIG = "1x4"  # 1 row, 4 columns of pillars


def _bin_to_grid(x_m: np.ndarray, y_m: np.ndarray, c: np.ndarray, nx: int, ny: int):
    x_mm = x_m * 1000.0
    y_um = y_m * 1e6
    x_edges = np.linspace(0.0, L_MM, nx + 1)
    y_edges = np.linspace(0.0, W_UM, ny + 1)
    sum_, _, _ = np.histogram2d(x_mm, y_um, bins=[x_edges, y_edges], weights=c)
    cnt, _, _ = np.histogram2d(x_mm, y_um, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, grid


def main() -> None:
    centres = read_cell_centres(CASE)
    x_m, y_m, _ = centres.T
    latest = find_latest_time(CASE)
    if latest is None:
        raise RuntimeError(f"no time directory in {CASE}")
    C = read_scalar_field(latest / "T")

    print(f"[info] using time directory {latest.name}")
    print(f"[info] {C.size} scalar values, {x_m.size} cell centres")
    if C.size != x_m.size:
        raise RuntimeError("cell-count mismatch")

    nx_grid, ny_grid = 200, 60
    xc_mm, yc_um, C_grid = _bin_to_grid(x_m, y_m, C, nx_grid, ny_grid)

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.32)

    ax_field = fig.add_subplot(gs[0, 0])
    pc = ax_field.pcolormesh(
        xc_mm, yc_um, C_grid.T, shading="nearest", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax_field.set_xlabel("Streamwise position $x$ (mm)", fontsize=11)
    ax_field.set_ylabel("Transverse position $y$ ($\\mu$m)", fontsize=11)
    ax_field.set_title(
        "(a) Pillar winner field — ladder $\\times$ 1$\\times$4, $H=200~\\mu$m\n"
        "$L_2 = 0.057$, $R^2 = 0.992$, $W = 2121~\\mu$m",
        fontsize=10.5,
    )
    cb = fig.colorbar(pc, ax=ax_field, fraction=0.045, pad=0.02)
    cb.set_label("Normalised concentration $C$", fontsize=10)

    # Strip boundaries
    for k in range(1, N_STRIPS):
        ax_field.axhline(W_UM * k / N_STRIPS, color="white", lw=0.4, alpha=0.4)

    # Pillar outlines: 1x4 — 4 pillars across x at y = W/2, evenly spaced
    n_cols = 4
    pillar_xs_mm = np.linspace(L_MM/(2*n_cols), L_MM*(2*n_cols-1)/(2*n_cols), n_cols)
    for px_mm in pillar_xs_mm:
        circ = mpatches.Circle(
            (px_mm, W_UM / 2.0), radius=D_P_UM/2.0,
            facecolor="none", edgecolor="white", lw=1.2,
        )
        ax_field.add_patch(circ)
    ax_field.text(
        0.5, W_UM/2 - 80, "  pillar row (1$\\times$4, $d_p$=185, $s_p$=427 $\\mu$m)",
        color="white", fontsize=8, ha="left",
    )

    # Panel (b)
    ax_prof = fig.add_subplot(gs[0, 1])
    y_target_um = np.linspace(0, W_UM, 200)
    ax_prof.plot(y_target_um / W_UM, y_target_um, "k--", lw=1.4, label="Linear target")
    Cmean_y = np.nanmean(C_grid, axis=0)
    ax_prof.plot(Cmean_y, yc_um, color="C4", lw=2.0, label="Depth-avg $\\langle C\\rangle_x$")
    for x_mm, col in zip([1.0, 5.0, 9.0], ["C2", "C1", "C3"]):
        ix = int(np.argmin(np.abs(xc_mm - x_mm)))
        ax_prof.plot(C_grid[ix, :], yc_um, color=col, lw=1.0, alpha=0.85,
                     label=f"$x={x_mm:.0f}$ mm")
    ax_prof.set_xlim(-0.05, 1.05)
    ax_prof.set_ylim(0, W_UM)
    ax_prof.set_xlabel("Concentration $C$", fontsize=11)
    ax_prof.set_ylabel("Transverse position $y$ ($\\mu$m)", fontsize=11)
    ax_prof.set_title("(b) Linearity check", fontsize=10.5)
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc="lower right", fontsize=8, framealpha=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
