"""Figure (c) — Cross-topology Sobol total-effect indices.

Main panel: three two-inlet topologies (opposing, same_side_Y, asymmetric_lumen)
as bar groups, parameters (r_flow, W, Q_total, theta, delta_W) as series. Three
visual messages:
  F1 — r_flow universally dominant (tall bars across all three).
  F2 — theta consistently inactive (floor-noise everywhere).
  F3 — W and Q_total trade off as the secondary lever.

Inset: ladder Sobol — confirmatory note that the 2-D (W, Q_total) design space
is internally well-conditioned (Sigma S_T ~ 1.01).

Opposing is annotated as "surrogate overfit (Sigma S_T = 1.81)".

Source: bo_<topology>_none_H<H>/interpretability/summary.json under
        examples/tumor_chip_linear_gradient/data/results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "examples/tumor_chip_linear_gradient/data/results"
OUT = REPO / "poster/figures/paper_v2/fig_c_cross_topology_sobol.png"

TWO_INLET = [
    ("opposing", "Opposing", "C0"),
    ("same_side_Y", "Same-side Y", "C1"),
    ("asymmetric_lumen", "Asymmetric lumen", "C2"),
]
PARAM_ORDER = ["r_flow", "W", "Q_total", "theta", "delta_W"]
PARAM_LABELS = {
    "r_flow": "$r_{\\rm flow}$",
    "W": "$W$",
    "Q_total": "$Q_{\\rm total}$",
    "theta": "$\\theta$",
    "delta_W": "$\\delta W$",
}


def _load(topology: str, H: int) -> dict:
    p = RES / f"bo_{topology}_none_H{H}/interpretability/summary.json"
    with p.open() as f:
        return json.load(f)


def _st_per_param(summary: dict) -> dict:
    names = summary["sobol"]["names"]
    st = summary["sobol"]["ST"]
    return dict(zip(names, st))


def main() -> None:
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])

    n_groups = len(PARAM_ORDER)
    width = 0.25
    xs = np.arange(n_groups)

    sigma_st = {}
    for j, (topology, label, color) in enumerate(TWO_INLET):
        s = _load(topology, 200)
        st_map = _st_per_param(s)
        bars = [st_map.get(p, 0.0) for p in PARAM_ORDER]
        sigma_st[topology] = sum(s["sobol"]["ST"])
        ax.bar(xs + (j - 1) * width, bars, width=width, color=color, label=label,
               edgecolor="black", linewidth=0.4)

    # Annotate r_flow bars (the F1 spotlight)
    for j, (topology, _, _) in enumerate(TWO_INLET):
        s = _load(topology, 200)
        st_map = _st_per_param(s)
        st_rflow = st_map.get("r_flow", 0.0)
        ax.text(
            xs[0] + (j - 1) * width, st_rflow + 0.02, f"{st_rflow:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([PARAM_LABELS[p] for p in PARAM_ORDER], fontsize=12)
    ax.set_ylabel("Total-effect Sobol index $S_T$", fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.axhline(0.0, color="black", lw=0.4)
    ax.grid(True, axis="y", alpha=0.3)

    legend_txts = []
    for topology, label, _ in TWO_INLET:
        flag = " ($\\Sigma S_T$={:.2f}, overfit)".format(sigma_st[topology]) \
            if topology == "opposing" else " ($\\Sigma S_T$={:.2f})".format(sigma_st[topology])
        legend_txts.append(label + flag)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_txts, loc="upper left",
              bbox_to_anchor=(0.32, 1.0), fontsize=9, framealpha=0.95)

    ax.set_title(
        "(a) Two-inlet trio ($H=200$ $\\mu$m, axis = $y$)",
        fontsize=12, pad=10,
    )

    # F1/F2/F3 spotlight banners — kept inside the plotting area, away from title
    ax.text(
        0, 1.05, "F1 — $r_{\\rm flow}$ universal driver",
        ha="center", fontsize=9, color="darkred",
    )
    ax.annotate(
        "F2 — $\\theta$ inactive\n(floor noise)",
        xy=(3, 0.04), xytext=(3, 0.32),
        ha="center", fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )
    ax.text(
        1.5, 0.55, "F3 — $W$ ⇄ $Q_{\\rm total}$\nsecondary lever",
        ha="center", fontsize=9, color="darkred",
    )

    # ---- Panel (b) — Ladder confirmatory ----
    inset = fig.add_subplot(gs[0, 1])
    s_ladder_h300 = _load("ladder", 300)
    s_ladder_h200 = _load("ladder", 200)
    ladder_params = s_ladder_h300["sobol"]["names"]
    st_h200 = s_ladder_h200["sobol"]["ST"]
    st_h300 = s_ladder_h300["sobol"]["ST"]
    xs_l = np.arange(len(ladder_params))
    inset.bar(xs_l - 0.18, st_h200, width=0.36, color="C3",
              edgecolor="black", linewidth=0.4, label="H=200")
    inset.bar(xs_l + 0.18, st_h300, width=0.36, color="C4",
              edgecolor="black", linewidth=0.4, label="H=300")
    for k, (a, b) in enumerate(zip(st_h200, st_h300)):
        inset.text(xs_l[k] - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        inset.text(xs_l[k] + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    inset.set_xticks(xs_l)
    inset.set_xticklabels([PARAM_LABELS[p] for p in ladder_params], fontsize=12)
    inset.set_ylabel("$S_T$", fontsize=11)
    inset.set_ylim(0.0, 1.05)
    inset.set_title(
        "(b) Ladder confirmatory (2-D)\n$\\Sigma S_T \\approx 1.01$",
        fontsize=11, pad=10,
    )
    inset.legend(fontsize=9, loc="upper left")
    inset.grid(True, axis="y", alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
