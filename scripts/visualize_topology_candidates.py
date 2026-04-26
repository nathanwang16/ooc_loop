"""
2-D top-down schematic renders of each topology candidate proposed in
examples/tumor_chip_linear_gradient/data/results/diagnostic_findings.md.

Rationale: chamber aspect ratio is L:W:H = 10000:3000:200, so 3-D matplotlib
renders are dominated by horizontal extent and z disappears unless artificially
exaggerated. Microfluidics literature universally uses top-down 2-D schematics
(Jeon 2000, Dertinger 2001, Whitesides 2008). We follow that convention.

Each panel: chamber footprint (light grey), inlet patches coloured by prescribed
concentration (blue C=0 → red C=1), outlet (green), auxiliary features
(side ports, membrane reservoir overlay, mixer tree) labelled at scale.

Output: examples/tumor_chip_linear_gradient/data/figures/topology_candidates/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "figures" / "topology_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chamber dimensions (μm) — pulled from default_config.yaml mid-bounds for visual clarity
L = 10000.0   # chamber length (x)
W = 3000.0    # chamber width (y)
H = 200.0     # chamber height (z), out of plane in this view

CMAP = LinearSegmentedColormap.from_list("conc", ["#2b6cb0", "#f7fafc", "#c53030"])  # blue→white→red


def _new_axes(title: str, xlim, ylim, *, height=4.6):
    fig, ax = plt.subplots(figsize=(11.5, height))
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("x  (flow direction, μm)", fontsize=10, labelpad=6)
    ax.set_ylabel("y  (μm)", fontsize=10, labelpad=10)
    ax.tick_params(labelsize=8, pad=3)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax


def _draw_chamber(ax, x0=0.0, x1=L, y0=0.0, y1=W):
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    facecolor="#e5e7eb", edgecolor="#374151",
                                    linewidth=1.2, zorder=1))


def _add_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cb.set_label("inlet concentration  C", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _save(fig, name: str):
    out = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Candidate A — Y-stacked N-inlet ladder (production winner)
# ---------------------------------------------------------------------------

def render_ladder(N: int = 8):
    fig, ax = _new_axes(f"A. Y-stacked ladder  (N={N} inlets)  →  axis=y linear gradient   [WINNER]",
                        xlim=(-2400, L + 1400), ylim=(-450, W + 450))
    _draw_chamber(ax)

    inlet_w = 250.0  # px in x for inlet patch
    dy = W / N
    for k in range(N):
        c = (k + 0.5) / N  # midpoint convention (production)
        ax.add_patch(mpatches.Rectangle((-inlet_w, k * dy), inlet_w, dy,
                                        facecolor=CMAP(c), edgecolor="#1f2937",
                                        linewidth=0.6, zorder=2))
    ax.text(-inlet_w - 250, W / 2, "N inlet strips\n(stacked in y)",
            ha="right", va="center", fontsize=9, color="#374151")

    # Outlet
    ax.add_patch(mpatches.Rectangle((L, 0), 250, W, facecolor="#22c55e",
                                    edgecolor="#15803d", alpha=0.75, zorder=2))
    ax.text(L + 400, W / 2, "outlet", fontsize=10, color="#15803d", va="center")

    # Flow arrow
    ax.annotate("", xy=(L * 0.55, W * 0.5), xytext=(L * 0.10, W * 0.5),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    ax.text(L * 0.32, W * 0.58, "advection (+x)", fontsize=9, color="#374151")

    _add_colorbar(fig, ax)
    _save(fig, "A_ladder_N8")


# ---------------------------------------------------------------------------
# Candidate B — Christmas-tree pre-mixer + chamber
# ---------------------------------------------------------------------------

def render_christmas_tree():
    tree_x_start = -8000.0
    tree_x_end = -200.0

    fig, ax = _new_axes("B. Christmas-tree pre-mixer + chamber  →  axis=y linear gradient",
                        xlim=(tree_x_start - 1700, L + 1700), ylim=(-W * 0.25, W * 1.15),
                        height=5.2)
    _draw_chamber(ax)

    # Two reagent reservoirs at far left
    res_y = [W * 0.25, W * 0.75]
    for y, c in zip(res_y, [0.0, 1.0]):
        ax.add_patch(mpatches.Rectangle((tree_x_start - 800, y - 200), 800, 400,
                                        facecolor=CMAP(c), edgecolor="#1f2937", lw=0.7))
        ax.text(tree_x_start - 400, y, f"C={c:.0f}", ha="center", va="center",
                fontsize=9, fontweight="bold")

    # Binary mixer tree: 2 → 3 → 5 → 8 streams (schematic — Whitesides 2000 style)
    levels = [2, 3, 5, 8]
    x_stations = np.linspace(tree_x_start, tree_x_end, len(levels) + 1)
    for i, n in enumerate(levels):
        x = x_stations[i + 1]
        ys = np.linspace(W * 0.1, W * 0.9, n)
        cs = np.linspace(0, 1, n)
        for y, c in zip(ys, cs):
            ax.add_patch(mpatches.Circle((x, y), 110, facecolor=CMAP(c),
                                         edgecolor="#1f2937", lw=0.6, zorder=3))
        if i + 1 < len(levels):
            ys_next = np.linspace(W * 0.1, W * 0.9, levels[i + 1])
            for y in ys:
                near = sorted(ys_next, key=lambda yy: abs(yy - y))[:2]
                for yn in near:
                    ax.plot([x, x_stations[i + 2]], [y, yn],
                            color="#6b7280", lw=0.8, alpha=0.7, zorder=2)

    # Final 8 strip-inlets at chamber x=0
    N = 8
    inlet_w = 200.0
    dy = W / N
    for k in range(N):
        c = (k + 0.5) / N
        ax.add_patch(mpatches.Rectangle((-inlet_w, k * dy), inlet_w, dy,
                                        facecolor=CMAP(c), edgecolor="#1f2937",
                                        lw=0.6, zorder=4))

    ax.add_patch(mpatches.Rectangle((L, 0), 250, W, facecolor="#22c55e",
                                    edgecolor="#15803d", alpha=0.75))
    ax.text(L + 400, W / 2, "outlet", fontsize=10, color="#15803d", va="center")

    ax.text(tree_x_start - 400, -W * 0.18, "2 reagent inputs", fontsize=9, color="#374151")
    ax.text((tree_x_start + tree_x_end) / 2, W * 1.05,
            "binary serpentine mixer tree (2 → 3 → 5 → 8 streams)",
            fontsize=9, color="#374151", ha="center")

    _add_colorbar(fig, ax)
    _save(fig, "B_christmas_tree")


# ---------------------------------------------------------------------------
# Candidate C — Distributed side-injection array  (only path to axis=x)
# ---------------------------------------------------------------------------

def render_side_injection(K: int = 8):
    fig, ax = _new_axes(f"C. Distributed side-injection  (K={K} drug ports along y=0)  →  axis=x linear gradient",
                        xlim=(-2400, L + 1400), ylim=(-W * 0.55, W + 450))
    _draw_chamber(ax)

    # Main medium inlet at x=0, full width (C=0)
    ax.add_patch(mpatches.Rectangle((-250, 0), 250, W, facecolor=CMAP(0.0),
                                    edgecolor="#1f2937", lw=0.6))
    ax.text(-500, W / 2, "medium\nC=0", ha="right", va="center", fontsize=9,
            color="#1e3a8a", fontweight="bold")

    # K side ports along y=0 with x-varying flow rate Q_k
    port_w = L / (K * 3.5)
    port_h = 220.0  # extends slightly into chamber for visibility
    for k in range(K):
        xc = (k + 1) * L / (K + 1)
        x0 = xc - port_w / 2
        ax.add_patch(mpatches.Rectangle((x0, -port_h), port_w, port_h,
                                        facecolor=CMAP(1.0), edgecolor="#7f1d1d",
                                        lw=0.6, zorder=3))
        # Q_k arrow into chamber
        ax.annotate("", xy=(xc, 80), xytext=(xc, -port_h * 0.3),
                    arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0))
        q_frac = (k + 1) / K
        ax.text(xc, -port_h - 240, f"Q$_{{{k+1}}}$={q_frac:.2f}",
                ha="center", fontsize=7.5, color="#7f1d1d")

    ax.text(L / 2, -W * 0.5, "drug ports along y=0 wall, increasing $Q_k \\propto k$",
            fontsize=9, color="#7f1d1d", ha="center")

    ax.add_patch(mpatches.Rectangle((L, 0), 250, W, facecolor="#22c55e",
                                    edgecolor="#15803d", alpha=0.75))
    ax.text(L + 400, W / 2, "outlet", fontsize=10, color="#15803d", va="center")

    # Flow arrow
    ax.annotate("", xy=(L * 0.55, W * 0.85), xytext=(L * 0.10, W * 0.85),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    ax.text(L * 0.32, W * 0.92, "advection (+x)", fontsize=9, color="#374151")

    _add_colorbar(fig, ax)
    _save(fig, "C_side_injection_K8")


# ---------------------------------------------------------------------------
# Candidate D — Permeable-floor membrane + reservoir
# ---------------------------------------------------------------------------

def render_permeable_wall():
    fig, ax = _new_axes("D. Permeable floor membrane + reservoir  →  axis=x linear gradient",
                        xlim=(-2400, L + 1400), ylim=(-450, W + 700))
    _draw_chamber(ax)

    # Main medium inlet (C=0)
    ax.add_patch(mpatches.Rectangle((-250, 0), 250, W, facecolor=CMAP(0.0),
                                    edgecolor="#1f2937", lw=0.6))
    ax.text(-500, W / 2, "medium\nC=0", ha="right", va="center", fontsize=9,
            color="#1e3a8a", fontweight="bold")

    # Permeable floor strips (top-down: permeability gradient shown as colour)
    n_strips = 12
    for i in range(n_strips):
        x0 = i * L / n_strips
        x1 = (i + 1) * L / n_strips
        permeability = (i + 0.5) / n_strips
        # Render as hatched overlay across full chamber width to indicate floor membrane
        ax.add_patch(mpatches.Rectangle((x0, 0), x1 - x0, W,
                                        facecolor=CMAP(permeability * 0.7 + 0.15),
                                        edgecolor="none", alpha=0.45, zorder=1.5))

    ax.text(L / 2, W + 250,
            "graded-permeability PDMS floor: $K(x) = K_0 \\cdot x/L$\n(reservoir at C=1 below the floor, out of plane)",
            fontsize=9, color="#374151", ha="center")

    ax.add_patch(mpatches.Rectangle((L, 0), 250, W, facecolor="#22c55e",
                                    edgecolor="#15803d", alpha=0.75))
    ax.text(L + 400, W / 2, "outlet", fontsize=10, color="#15803d", va="center")

    ax.annotate("", xy=(L * 0.55, W * 0.5), xytext=(L * 0.10, W * 0.5),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    ax.text(L * 0.32, W * 0.58, "advection (+x)", fontsize=9, color="#374151")

    _add_colorbar(fig, ax)
    _save(fig, "D_permeable_wall")


# ---------------------------------------------------------------------------
# Candidate E — Counter-flow inlets (rejected — unsteady risk)
# ---------------------------------------------------------------------------

def render_counter_flow():
    fig, ax = _new_axes("E. Counter-flow inlets  →  axis=x  (REJECTED — unsteady-flow risk)",
                        xlim=(-2400, L + 2400), ylim=(-W * 0.3, W * 1.55))
    _draw_chamber(ax)

    # Drug inlet at x=0 (small, low Q, C=1)
    drug_y0 = W * 0.4
    drug_y1 = W * 0.6
    ax.add_patch(mpatches.Rectangle((-250, drug_y0), 250, drug_y1 - drug_y0,
                                    facecolor=CMAP(1.0), edgecolor="#7f1d1d", lw=0.7))
    ax.text(-500, W * 0.5, "drug\nC=1\n(low Q)", ha="right", va="center",
            fontsize=8.5, color="#7f1d1d", fontweight="bold")

    # Medium inlet at x=L (full face, high Q, C=0)
    ax.add_patch(mpatches.Rectangle((L, 0), 250, W, facecolor=CMAP(0.0),
                                    edgecolor="#1e3a8a", lw=0.7))
    ax.text(L + 500, W / 2, "medium\nC=0\n(high Q)", ha="left", va="center",
            fontsize=8.5, color="#1e3a8a", fontweight="bold")

    # Side outlets at y=0 and y=W, mid-chamber
    for y_wall, y_off in ((0, -220), (W, 0)):
        x0 = L * 0.4
        x1 = L * 0.6
        ax.add_patch(mpatches.Rectangle((x0, y_off + (y_wall - 0)), x1 - x0, 220,
                                        facecolor="#22c55e", edgecolor="#15803d", alpha=0.75))
    ax.text(L * 0.5, W + 750, "opposing flows; stagnation surface mid-chamber",
            fontsize=9, color="#374151", ha="center")
    ax.text(L * 0.5, W + 350, "side outlets at y=0 and y=W (mid-chamber)",
            fontsize=9, color="#15803d", ha="center")

    # Counter-flow arrows
    ax.annotate("", xy=(L * 0.45, W * 0.5), xytext=(L * 0.05, W * 0.5),
                arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.6))
    ax.annotate("", xy=(L * 0.55, W * 0.5), xytext=(L * 0.95, W * 0.5),
                arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.6))

    _add_colorbar(fig, ax)
    _save(fig, "E_counter_flow")


def main() -> int:
    print(f"Writing topology schematics to {OUT_DIR.relative_to(REPO_ROOT)}/")
    render_ladder(N=8)
    render_christmas_tree()
    render_side_injection(K=8)
    render_permeable_wall()
    render_counter_flow()
    print("\nView them with:")
    print(f"  open {OUT_DIR.relative_to(REPO_ROOT)}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
