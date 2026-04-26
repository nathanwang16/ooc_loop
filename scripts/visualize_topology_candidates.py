"""
Schematic 3-D renders of each topology candidate proposed in
examples/tumor_chip_linear_gradient/data/results/diagnostic_findings.md.

Produces one PNG per candidate showing chamber walls (semi-transparent),
inlet patches (coloured by prescribed concentration: blue C=0 → red C=1),
outlet patches (green), and any auxiliary features (membrane, reservoir,
mixer tree).  These are *schematics*, not meshable geometry — they exist
to give a quick visual intuition before implementation.

Output: examples/tumor_chip_linear_gradient/data/figures/topology_candidates/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "figures" / "topology_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chamber dimensions (μm) — pulled from default_config.yaml mid-bounds for visual clarity
L = 10000.0   # chamber length (x)
W = 3000.0    # chamber width (y)
H = 200.0     # chamber height (z)

CMAP = LinearSegmentedColormap.from_list("conc", ["#2b6cb0", "#f7fafc", "#c53030"])  # blue→white→red


def _new_axes(title: str):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [μm]  (flow direction)")
    ax.set_ylabel("y [μm]")
    ax.set_zlabel("z [μm]")
    ax.set_box_aspect([L, W, H * 6])  # exaggerate z for visibility
    return fig, ax


def _draw_chamber(ax, L=L, W=W, H=H, alpha=0.06):
    """Translucent chamber box."""
    verts = [
        [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)],          # floor
        [(0, 0, H), (L, 0, H), (L, W, H), (0, W, H)],          # ceiling
        [(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)],          # x=0 face
        [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)],          # x=L face
        [(0, 0, 0), (L, 0, 0), (L, 0, H), (0, 0, H)],          # y=0 wall
        [(0, W, 0), (L, W, 0), (L, W, H), (0, W, H)],          # y=W wall
    ]
    pc = Poly3DCollection(verts, facecolor="#9ca3af", edgecolor="#374151",
                          linewidth=0.4, alpha=alpha)
    ax.add_collection3d(pc)


def _patch(ax, verts, color, alpha=0.95, edge="#1f2937"):
    pc = Poly3DCollection([verts], facecolor=color, edgecolor=edge, linewidth=0.5, alpha=alpha)
    ax.add_collection3d(pc)


def _save(fig, name: str):
    out = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Candidate A — Y-stacked N-inlet ladder
# ---------------------------------------------------------------------------

def render_ladder(N: int = 8):
    fig, ax = _new_axes(f"A. Y-stacked ladder (N={N} inlets)  →  axis=y linear gradient")
    _draw_chamber(ax)

    dy = W / N
    for k in range(N):
        c = k / (N - 1)  # prescribed concentration
        color = CMAP(c)
        y0 = k * dy
        y1 = (k + 1) * dy
        # Inlet patch on x=0 face (full z)
        verts = [(0, y0, 0), (0, y1, 0), (0, y1, H), (0, y0, H)]
        _patch(ax, verts, color, alpha=0.95)
        ax.text(-200, (y0 + y1) / 2, H / 2, f"C={c:.2f}", fontsize=7, ha="right", va="center")

    # Outlet at x=L face — green
    out_verts = [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)]
    _patch(ax, out_verts, "#22c55e", alpha=0.55)
    ax.text(L + 200, W / 2, H / 2, "outlet", fontsize=8, color="#15803d")

    ax.view_init(elev=22, azim=-55)
    _save(fig, "A_ladder_N8")


# ---------------------------------------------------------------------------
# Candidate B — Christmas-tree pre-mixer + chamber
# ---------------------------------------------------------------------------

def render_christmas_tree():
    """Top-down (z-flat) schematic since the tree is essentially 2-D."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_title("B. Christmas-tree pre-mixer + chamber  →  axis=y linear gradient", fontsize=11)
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_aspect("equal")

    # Tree network on the LEFT of the chamber (negative x)
    tree_x_start = -8000.0
    tree_x_end = 0.0

    # 2 reservoirs at x=tree_x_start
    res_y = [W * 0.25, W * 0.75]
    for y, c in zip(res_y, [0.0, 1.0]):
        ax.add_patch(plt.Rectangle((tree_x_start - 800, y - 200), 800, 400,
                                    facecolor=CMAP(c), edgecolor="#1f2937"))
        ax.text(tree_x_start - 400, y, f"C={c:.0f}", ha="center", va="center", fontsize=8)

    # Binary tree of mixers — 3 levels: 2 → 4 → 8
    levels = [2, 3, 5, 8]   # number of streams at each x-station
    x_stations = np.linspace(tree_x_start, tree_x_end, len(levels) + 1)
    for i, n in enumerate(levels):
        x = x_stations[i + 1]
        ys = np.linspace(W * 0.1, W * 0.9, n)
        cs = np.linspace(0, 1, n)
        for y, c in zip(ys, cs):
            ax.add_patch(plt.Circle((x, y), 80, facecolor=CMAP(c), edgecolor="#1f2937", lw=0.5))
        # serpentine connection lines (schematic)
        if i + 1 < len(levels):
            ys_next = np.linspace(W * 0.1, W * 0.9, levels[i + 1])
            for y in ys:
                # connect each node to nearest 2 in next station
                near = sorted(ys_next, key=lambda yy: abs(yy - y))[:2]
                for yn in near:
                    ax.plot([x, x_stations[i + 2]], [y, yn], color="#6b7280", lw=0.5, alpha=0.6)

    # Chamber rectangle
    ax.add_patch(plt.Rectangle((0, 0), L, W, facecolor="#9ca3af", edgecolor="#374151", alpha=0.15))
    # Inlet patches at x=0 — 8 strips, ladder-coloured
    N = 8
    dy = W / N
    for k in range(N):
        c = k / (N - 1)
        ax.add_patch(plt.Rectangle((0, k * dy), 80, dy, facecolor=CMAP(c), edgecolor="#1f2937", lw=0.5))
    ax.add_patch(plt.Rectangle((L - 80, 0), 80, W, facecolor="#22c55e", alpha=0.6, edgecolor="#15803d"))
    ax.text(L + 200, W / 2, "outlet", fontsize=8, color="#15803d", va="center")

    ax.text(tree_x_start - 1500, -W * 0.15, "2 reagent inputs", fontsize=8, color="#374151")
    ax.text(L / 2, -W * 0.15, "wide chamber (gradient flows in +x)", fontsize=8, color="#374151", ha="center")

    ax.set_xlim(tree_x_start - 1500, L + 1500)
    ax.set_ylim(-W * 0.25, W * 1.15)
    _save(fig, "B_christmas_tree")


# ---------------------------------------------------------------------------
# Candidate C — Distributed side-injection array
# ---------------------------------------------------------------------------

def render_side_injection(K: int = 8):
    fig, ax = _new_axes(f"C. Distributed side-injection (K={K} drug ports)  →  axis=x linear gradient")
    _draw_chamber(ax)

    # Main medium inlet — full x=0 face, blue (C=0)
    main_inlet = [(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)]
    _patch(ax, main_inlet, CMAP(0.0), alpha=0.85)
    ax.text(-300, W / 2, H / 2, "medium\nC=0", fontsize=8, ha="right", va="center")

    # K small drug ports along y=0 wall, ramping flow rate with x
    port_w = L / (K * 3)        # port length in x
    port_z_extent = H * 0.6     # ports occupy lower portion of z for visualisation
    for k in range(K):
        xc = (k + 1) * L / (K + 1)
        x0 = xc - port_w / 2
        x1 = xc + port_w / 2
        # Port on y=0 wall
        verts = [(x0, 0, H * 0.2), (x1, 0, H * 0.2), (x1, 0, H * 0.2 + port_z_extent), (x0, 0, H * 0.2 + port_z_extent)]
        _patch(ax, verts, CMAP(1.0), alpha=0.9, edge="#7f1d1d")
        # Flow-rate annotation (rises with x)
        q_frac = (k + 1) / K
        ax.text(xc, -200, H * 1.2, f"Q={q_frac:.2f}", fontsize=6.5, ha="center", color="#7f1d1d")

    ax.text(L / 2, -1100, H * 1.6, f"{K} drug ports along y=0 wall, increasing Q with x", fontsize=8, ha="center", color="#7f1d1d")

    # Outlet
    out_verts = [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)]
    _patch(ax, out_verts, "#22c55e", alpha=0.55)
    ax.text(L + 200, W / 2, H / 2, "outlet", fontsize=8, color="#15803d")

    ax.view_init(elev=24, azim=-55)
    _save(fig, "C_side_injection_K8")


# ---------------------------------------------------------------------------
# Candidate D — Permeable-wall drip with reservoir
# ---------------------------------------------------------------------------

def render_permeable_wall():
    fig, ax = _new_axes("D. Permeable floor membrane + reservoir  →  axis=x linear gradient")

    # Reservoir below floor (negative z)
    H_res = H * 1.5
    res_verts = [
        [(0, 0, -H_res), (L, 0, -H_res), (L, W, -H_res), (0, W, -H_res)],
        [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)],
        [(0, 0, -H_res), (0, W, -H_res), (0, W, 0), (0, 0, 0)],
        [(L, 0, -H_res), (L, W, -H_res), (L, W, 0), (L, 0, 0)],
        [(0, 0, -H_res), (L, 0, -H_res), (L, 0, 0), (0, 0, 0)],
        [(0, W, -H_res), (L, W, -H_res), (L, W, 0), (0, W, 0)],
    ]
    pc = Poly3DCollection(res_verts, facecolor=CMAP(1.0), edgecolor="#7f1d1d", linewidth=0.3, alpha=0.18)
    ax.add_collection3d(pc)
    ax.text(L / 2, W / 2, -H_res * 0.7, "reservoir C=1", ha="center", fontsize=8, color="#7f1d1d")

    # Permeable floor — gradient-coloured to indicate increasing permeability with x
    n_strips = 12
    for i in range(n_strips):
        x0 = i * L / n_strips
        x1 = (i + 1) * L / n_strips
        permeability = (i + 0.5) / n_strips  # 0..1
        verts = [(x0, 0, 0), (x1, 0, 0), (x1, W, 0), (x0, W, 0)]
        _patch(ax, verts, CMAP(permeability * 0.6 + 0.2), alpha=0.6, edge="#374151")

    ax.text(L / 2, W * 1.15, H * 0.8, "graded permeability K(x) = K₀·x/L", fontsize=8, ha="center", color="#374151")

    # Inlet (medium, C=0)
    in_verts = [(0, 0, 0), (0, W, 0), (0, W, H), (0, 0, H)]
    _patch(ax, in_verts, CMAP(0.0), alpha=0.85)
    ax.text(-300, W / 2, H / 2, "medium\nC=0", fontsize=8, ha="right", va="center")

    # Outlet
    out_verts = [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)]
    _patch(ax, out_verts, "#22c55e", alpha=0.55)
    ax.text(L + 200, W / 2, H / 2, "outlet", fontsize=8, color="#15803d")

    ax.set_box_aspect([L, W, (H + H_res) * 4])
    ax.view_init(elev=20, azim=-60)
    _save(fig, "D_permeable_wall")


# ---------------------------------------------------------------------------
# Candidate E — Counter-flow inlets
# ---------------------------------------------------------------------------

def render_counter_flow():
    fig, ax = _new_axes("E. Counter-flow inlets  →  axis=x (weak gradient; unsteady risk)")
    _draw_chamber(ax)

    # Drug inlet at x=0, small
    in_drug = [(0, W * 0.4, 0), (0, W * 0.6, 0), (0, W * 0.6, H), (0, W * 0.4, H)]
    _patch(ax, in_drug, CMAP(1.0), alpha=0.9)
    ax.text(-300, W * 0.5, H / 2, "drug\nC=1\n(low Q)", fontsize=7, ha="right", va="center", color="#7f1d1d")

    # Medium inlet at x=L, large (full face)
    in_med = [(L, 0, 0), (L, W, 0), (L, W, H), (L, 0, H)]
    _patch(ax, in_med, CMAP(0.0), alpha=0.85)
    ax.text(L + 300, W / 2, H / 2, "medium\nC=0\n(high Q)", fontsize=7, ha="left", va="center", color="#1e3a8a")

    # Outlets on y=0 and y=W mid-chamber
    for y_wall in (0, W):
        x0 = L * 0.4
        x1 = L * 0.6
        verts = [(x0, y_wall, 0), (x1, y_wall, 0), (x1, y_wall, H), (x0, y_wall, H)]
        _patch(ax, verts, "#22c55e", alpha=0.6)
    ax.text(L * 0.5, -250, H * 1.3, "outlets on side walls", fontsize=8, ha="center", color="#15803d")

    # Arrows showing counter-flow
    ax.quiver(L * 0.05, W / 2, H * 1.3, L * 0.15, 0, 0, color="#7f1d1d", arrow_length_ratio=0.3, lw=1.5)
    ax.quiver(L * 0.95, W / 2, H * 1.3, -L * 0.15, 0, 0, color="#1e3a8a", arrow_length_ratio=0.3, lw=1.5)

    ax.view_init(elev=22, azim=-55)
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
    print("Or in your image viewer of choice (Preview, IINA, etc.).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
