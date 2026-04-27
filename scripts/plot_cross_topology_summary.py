"""Cross-topology BO summary bar chart for the poster.

Reads best-feasible L2 from each BO state directory and renders a horizontal
bar chart with the uniform-field reference line.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RES = Path("examples/tumor_chip_linear_gradient/data/results")

RUNS = [
    ("asymmetric_lumen", "bo_asymmetric_lumen_none_H200"),
    ("same_side_Y",      "bo_same_side_Y_none_H200"),
    ("opposing",         "bo_opposing_none_H200"),
    ("ladder (H=200)",   "bo_ladder_none_H200"),
    ("ladder (H=300)",   "bo_ladder_none_H300"),
]


def best_feasible_l2(state_dir: Path) -> float:
    payload = json.loads((state_dir / "evaluations.json").read_text())
    feas = [r for r in payload["evaluations"] if r.get("feasible")]
    if not feas:
        raise RuntimeError(f"No feasible records in {state_dir}")
    return min(r["metrics"]["L2_to_target"] for r in feas)


def main(out_path: Path) -> None:
    labels: list[str] = []
    vals: list[float] = []
    for label, dirname in RUNS:
        l2 = best_feasible_l2(RES / dirname)
        labels.append(label)
        vals.append(l2)
        print(f"{label:25s}  L2={l2:.4f}")

    colors = ["#6b7280", "#6b7280", "#6b7280", "#dc2626", "#16a34a"]
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    bars = ax.barh(labels, vals, color=colors, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.018, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=15)
    ax.axvline(0.585, ls="--", color="#9ca3af", lw=2.0,
               label="uniform-field floor (axis=x, mass-conservation argument)")
    ax.set_xlabel("Best feasible normalised-RMS $L_2$ to target  (lower is better)",
                  fontsize=16, labelpad=8)
    ax.set_title("Cross-topology BO winners — linear y-axis gradient target",
                 fontsize=18, pad=12)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(RES / "cross_topology_summary.png")
