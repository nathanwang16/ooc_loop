"""Figure (g) — Ladder Sobol regime swap (pillars=none vs pillars=1x4).

Side-by-side bar chart of total-effect Sobol indices for the same ladder
topology under two pillar configurations:
  Left  panel: pillars=none, 2-D BO over (W, Q_total). Q_total dominates.
  Right panel: pillars=1x4, 4-D BO over (W, d_p, s_p, Q_total). W dominates,
                s_p emerges as second knob, Q_total collapses to ~0.

This is the most surprising finding of the project: the same chamber inverts
which parameter controls the gradient depending on whether obstacles are
present. Caveats: the pillar winner sits at tau ~2.6 Pa, above the 2.0 Pa cap;
the absolute-best feasible under the production constraint set is 0.0588.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NONE_SUM = REPO / "examples/tumor_chip_linear_gradient/data/results/bo_ladder_none_H200/interpretability/summary.json"
PILLAR_SUM = REPO / "examples/tumor_chip_linear_gradient/data/results_ladder_pillar_ablation/bo_ladder_1x4_H200/interpretability/summary.json"
OUT = REPO / "poster/figures/paper_v2/fig_g_pillar_regime_swap.png"

PARAM_LABELS = {
    "W": "$W$",
    "Q_total": "$Q_{\\rm total}$",
    "d_p": "$d_p$",
    "s_p": "$s_p$",
}


def load(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as f:
        s = json.load(f)
    return s["sobol"]["names"], np.array(s["sobol"]["ST"])


def main() -> None:
    none_names, none_st = load(NONE_SUM)
    pill_names, pill_st = load(PILLAR_SUM)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    # Left — pillars=none
    xs = np.arange(len(none_names))
    bars = ax1.bar(xs, none_st, color="C3", edgecolor="black", linewidth=0.4)
    for x, st in zip(xs, none_st):
        ax1.text(x, st + 0.02, f"{st:.2f}", ha="center", fontsize=9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([PARAM_LABELS.get(n, n) for n in none_names], fontsize=12)
    ax1.set_ylabel("Total-effect Sobol index $S_T$", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("(a) Ladder, pillars = none\n2-D, $\\Sigma S_T = 1.01$", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right — pillars=1x4
    xs2 = np.arange(len(pill_names))
    ax2.bar(xs2, pill_st, color="C4", edgecolor="black", linewidth=0.4)
    for x, st in zip(xs2, pill_st):
        ax2.text(x, st + 0.02, f"{st:.3f}" if st < 0.1 else f"{st:.2f}",
                 ha="center", fontsize=9)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([PARAM_LABELS.get(n, n) for n in pill_names], fontsize=12)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("(b) Ladder, pillars = 1$\\times$4\n4-D, $\\Sigma S_T = 1.00$", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    # Annotate the regime swap
    ax2.annotate(
        "$Q_{\\rm total}$ collapses\n($S_T$: 0.87 $\\to$ 0.001)",
        xy=(3, 0.05), xytext=(3.1, 0.50),
        ha="center", fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )
    ax2.annotate(
        "$s_p$ emerges as\nsecondary lever",
        xy=(2, 0.14), xytext=(1.5, 0.45),
        ha="center", fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )
    ax1.annotate(
        "$Q_{\\rm total}$ dominates\nresidence-time regime",
        xy=(1, 0.87), xytext=(0.5, 0.55),
        ha="center", fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )

    fig.suptitle(
        "Ladder Sobol regime swap: residence-time $\\to$ geometric-mixer when pillars are introduced",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
