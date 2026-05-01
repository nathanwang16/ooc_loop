"""Figure (b) — Constraint binding diagnostic.

Shows, for each topology's best-feasible design at H=200 (plus the ladder
H=300 winner), how close each of the 5 constraints is to its bound.
Slack is normalised so 0 = binding, 1 = far from bound. Read off:
  - At H=200 ladder, AR and tau_mean are both pinned (slack ≈ 0).
  - At H=300 ladder, AR and Q_total are pinned; tau_mean releases.
  - asymmetric_lumen is f_dead-bound (topology-intrinsic).

Source: optimization_summary_*.json + evaluations_*.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "examples/tumor_chip_linear_gradient/data/results"
OUT = REPO / "poster/figures/paper_v2/fig_b_constraint_binding.png"

# Constraint thresholds (from config.yaml)
TAU_MIN, TAU_MAX = 0.1, 2.0
F_DEAD_MAX = 0.08
RE_MAX = 100.0
AR_MAX = 15.0
Q_MAX = 200.0


def _is_feasible(rec: dict, H_um: float) -> bool:
    m = rec["metrics"]
    if not bool(m.get("converged", False)) or not bool(m.get("mesh_ok", True)):
        return False
    tau = float(m.get("tau_mean", 0.0))
    if not (TAU_MIN <= tau <= TAU_MAX):
        return False
    if float(m.get("f_dead", 1.0)) > F_DEAD_MAX:
        return False
    if float(m.get("Re", 1e9)) > RE_MAX:
        return False
    W_um = float(rec["params"]["W"])
    if W_um / H_um > AR_MAX + 1e-6:
        return False
    return True


def _best_feasible(top: str, H: int) -> dict:
    p = RES / f"evaluations_{top}_none_H{H}.jsonl"
    best, best_l2 = None, np.inf
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            if not _is_feasible(rec, H):
                continue
            l2 = float(rec["metrics"].get("L2_to_target", np.inf))
            if l2 < best_l2:
                best_l2, best = l2, rec
    return best


def _slack(value: float, threshold: float, *, kind: str) -> float:
    """Return [0, 1] slack: 0 = binding, 1 = far from bound."""
    if kind == "upper":
        if threshold <= 0:
            return 1.0
        return max(0.0, min(1.0, (threshold - value) / threshold))
    if kind == "lower":
        # symmetric reading; for tau_min we just report (value-thr) / (TAU_MAX-thr)
        return max(0.0, min(1.0, (value - threshold) / (TAU_MAX - threshold)))
    raise ValueError(kind)


def _normalised_value(rec: dict, H: float) -> dict:
    m = rec["metrics"]
    return {
        "tau_max": float(m["tau_mean"]) / TAU_MAX,
        "f_dead": float(m["f_dead"]) / F_DEAD_MAX,
        "Re": float(m["Re"]) / RE_MAX,
        "W/H": (float(rec["params"]["W"]) / H) / AR_MAX,
        "Q_total": float(rec["params"]["Q_total"]) / Q_MAX,
    }


def main() -> None:
    rows = [
        ("ladder", 300, "Ladder $H=300$", "C4"),
        ("ladder", 200, "Ladder $H=200$", "C3"),
        ("opposing", 200, "Opposing $H=200$", "C0"),
        ("same_side_Y", 200, "Same-side Y $H=200$", "C1"),
        ("asymmetric_lumen", 200, "Asym. lumen $H=200$", "C2"),
    ]
    constraints = ["W/H", "tau_max", "Q_total", "f_dead", "Re"]
    constraint_labels = {
        "W/H": "$W/H \\leq 15$",
        "tau_max": "$\\tau_{\\rm mean} \\leq 2.0$ Pa",
        "Q_total": "$Q_{\\rm total} \\leq 200$ $\\mu$L/min",
        "f_dead": "$f_{\\rm dead} \\leq 0.08$",
        "Re": "$\\rm Re \\leq 100$",
    }

    fig, ax = plt.subplots(figsize=(10, 5.0))

    n_rows = len(rows)
    n_cons = len(constraints)
    bar_h = 0.16
    ys = np.arange(n_cons)

    for j, (top, H, label, color) in enumerate(rows):
        rec = _best_feasible(top, H)
        if rec is None:
            print(f"[warn] no feasible record for {top} H={H}")
            continue
        nv = _normalised_value(rec, H)
        offsets = (j - (n_rows - 1) / 2) * bar_h
        # Plot value/threshold bar (1.0 = binding); shorter = more slack
        for i, c in enumerate(constraints):
            v = nv[c]
            ax.barh(
                ys[i] + offsets, v, height=bar_h * 0.9, color=color,
                edgecolor="black", linewidth=0.4,
                label=label if i == 0 else None,
            )
            if v >= 0.985:
                ax.text(v + 0.01, ys[i] + offsets, "BIND", fontsize=7,
                        va="center", color="darkred", fontweight="bold")

    ax.axvline(1.0, color="black", ls="-", lw=1.2)
    ax.text(1.005, n_cons - 0.4, "constraint", fontsize=8, color="black", rotation=90)
    ax.text(1.04, n_cons - 0.4, "bound", fontsize=8, color="black", rotation=90)
    ax.set_yticks(ys)
    ax.set_yticklabels([constraint_labels[c] for c in constraints], fontsize=10)
    ax.set_xlabel("Optimum value / constraint threshold (1.0 = binding)", fontsize=11)
    ax.set_xlim(0, 1.18)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(
        "Constraint-binding diagnostic at each topology's best-feasible design",
        fontsize=12, pad=8,
    )

    # De-duplicated legend (one entry per topology)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    ax.legend(h2, l2, loc="lower right", fontsize=8, framealpha=0.95, ncol=1)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
