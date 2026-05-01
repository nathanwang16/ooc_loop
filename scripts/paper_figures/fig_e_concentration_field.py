"""Figure (e) — H=300 ladder winner concentration field.

Two-panel figure:
  (a) 2-D heatmap of the converged C(x,y) field across the chamber.
  (b) Depth-averaged C(y) profile vs. the linear target, with a small
      multi-station overlay showing the gradient persistence at three
      downstream stations (x = 1, 5, 9 mm).

Source case directory:
    examples/tumor_chip_linear_gradient/data/cases/
        run_ladder_none_H300_1777238804412_67025_3420ef/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ooc_optimizer.cfd.foam_parser import (  # noqa: E402
    read_cell_centres,
    read_scalar_field,
)

CASE = REPO / (
    "examples/tumor_chip_linear_gradient/data/cases/"
    "run_ladder_none_H300_1777238804412_67025_3420ef"
)
LATEST_TIME = CASE / "881"  # converged time step
OUT = REPO / "poster/figures/paper_v2/fig_e_concentration_field.png"

# Winner parameters (from optimization_summary_ladder_H_sweep.json).
W_UM = 4495.6
H_UM = 300.0
L_MM = 10.0
N_STRIPS = 8


def _bin_to_grid(x_m: np.ndarray, y_m: np.ndarray, c: np.ndarray, nx: int, ny: int):
    """Bin scattered cell-centre values into a regular nx x ny grid for plotting."""
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
    C = read_scalar_field(LATEST_TIME / "T")

    if C.size != x_m.size:
        raise RuntimeError(
            f"Cell-count mismatch: {C.size} scalar values, {x_m.size} centres"
        )

    nx_grid, ny_grid = 200, 88
    xc_mm, yc_um, C_grid = _bin_to_grid(x_m, y_m, C, nx_grid, ny_grid)

    fig = plt.figure(figsize=(10.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.32)

    ax_field = fig.add_subplot(gs[0, 0])
    pc = ax_field.pcolormesh(
        xc_mm, yc_um, C_grid.T, shading="nearest", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax_field.set_xlabel("Streamwise position $x$ (mm)", fontsize=11)
    ax_field.set_ylabel("Transverse position $y$ ($\\mu$m)", fontsize=11)
    ax_field.set_title(
        "(a) Converged concentration field — H=300 $\\mu$m ladder winner\n"
        "$L_2 = 0.067$, $R^2 = 0.990$",
        fontsize=11,
    )
    ax_field.set_aspect("auto")
    cb = fig.colorbar(pc, ax=ax_field, fraction=0.045, pad=0.02)
    cb.set_label("Normalised concentration $C$", fontsize=10)

    # Overlay strip boundaries
    for k in range(1, N_STRIPS):
        ax_field.axhline(W_UM * k / N_STRIPS, color="white", lw=0.4, alpha=0.4)

    # ---- Panel (b): depth-averaged C(y) vs target, plus three stations ----
    ax_prof = fig.add_subplot(gs[0, 1])

    y_target_um = np.linspace(0, W_UM, 200)
    C_target = y_target_um / W_UM
    ax_prof.plot(C_target, y_target_um, "k--", lw=1.4, label="Linear target")

    # Depth-averaged across all x
    Cmean_y = np.nanmean(C_grid, axis=0)
    ax_prof.plot(Cmean_y, yc_um, color="C0", lw=2.0, label="Depth-averaged $\\langle C\\rangle_x$")

    # Three downstream stations
    station_mm = [1.0, 5.0, 9.0]
    colors = ["C2", "C1", "C3"]
    for x_mm, col in zip(station_mm, colors):
        ix = int(np.argmin(np.abs(xc_mm - x_mm)))
        ax_prof.plot(
            C_grid[ix, :], yc_um, color=col, lw=1.0, alpha=0.85,
            label=f"$x = {x_mm:.0f}$ mm",
        )

    ax_prof.set_xlim(-0.05, 1.05)
    ax_prof.set_ylim(0, W_UM)
    ax_prof.set_xlabel("Concentration $C$", fontsize=11)
    ax_prof.set_ylabel("Transverse position $y$ ($\\mu$m)", fontsize=11)
    ax_prof.set_title(
        "(b) Linearity check\n"
        "depth-averaged + 3 downstream stations",
        fontsize=11,
    )
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("", fontsize=12)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
