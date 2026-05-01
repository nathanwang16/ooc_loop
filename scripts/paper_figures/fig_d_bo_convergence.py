"""Figure (d) — BO convergence curves.

Two panels:
  (a) All four topologies at H=200 μm — best feasible L2 vs evaluation index.
      Shows the topology gap: ladder converges to ~0.082 within ~50 evals;
      two-inlet topologies plateau at ~0.88 / 0.99 / 1.09.
  (b) Ladder only — H=200 vs H=300 — shows the H=300 curve dropping below
      H=200 at the same iteration budget, plus the higher feasibility rate.

Source: evaluations_<topology>_none_H<H>.jsonl in
        examples/tumor_chip_linear_gradient/data/results/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "examples/tumor_chip_linear_gradient/data/results"
OUT = REPO / "poster/figures/paper_v2/fig_d_bo_convergence.png"

# Constraints — match bo_loop.py / config.yaml
TAU_MIN, TAU_MAX = 0.1, 2.0
F_DEAD_MAX = 0.08
RE_MAX = 100.0
AR_MAX = 15.0


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


def _load_curve(topology: str, H: int):
    p = RES / f"evaluations_{topology}_none_H{H}.jsonl"
    L2: list[float] = []
    feas: list[bool] = []
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            L2.append(float(rec["metrics"].get("L2_to_target", np.inf)))
            feas.append(_is_feasible(rec, H))
    L2 = np.array(L2)
    feas = np.array(feas)
    best = np.full(len(L2), np.nan)
    cur = np.inf
    for i in range(len(L2)):
        if feas[i] and np.isfinite(L2[i]):
            cur = min(cur, L2[i])
        best[i] = cur if np.isfinite(cur) else np.nan
    return np.arange(1, len(L2) + 1), best, feas


TOPOLOGIES_H200 = [
    ("ladder", "Ladder", "C3"),
    ("opposing", "Opposing", "C0"),
    ("same_side_Y", "Same-side Y", "C1"),
    ("asymmetric_lumen", "Asymmetric lumen", "C2"),
]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- Panel (a) — H=200 cross-topology ----
    for top, label, color in TOPOLOGIES_H200:
        x, best, feas = _load_curve(top, 200)
        ax1.plot(x, best, color=color, lw=1.6, label=f"{label} (n_feas={feas.sum()}/{len(feas)})")
        # Sobol init / BO boundary marker on first curve
    ax1.axvline(24, color="grey", ls=":", lw=0.8)
    ax1.text(24, 1.55, " Sobol\n init →", fontsize=8, color="grey", va="top")
    ax1.set_yscale("log")
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0.05, 2.0)
    ax1.set_xlabel("Evaluation index", fontsize=11)
    ax1.set_ylabel("Best feasible $L_2$ so far", fontsize=11)
    ax1.set_title("(a) Cross-topology BO convergence ($H = 200$ $\\mu$m)", fontsize=11)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # ---- Panel (b) — Ladder H=200 vs H=300 ----
    x200, best200, feas200 = _load_curve("ladder", 200)
    x300, best300, feas300 = _load_curve("ladder", 300)
    ax2.plot(x200, best200, color="C3", lw=1.8,
             label=f"Ladder $H=200$ (n_feas={feas200.sum()}/{len(feas200)})")
    ax2.plot(x300, best300, color="C4", lw=1.8,
             label=f"Ladder $H=300$ (n_feas={feas300.sum()}/{len(feas300)})")
    ax2.axhline(0.0817, color="C3", ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(0.0671, color="C4", ls="--", lw=0.8, alpha=0.7)
    ax2.axvline(24, color="grey", ls=":", lw=0.8)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0.06, 0.22)
    ax2.set_xlabel("Evaluation index", fontsize=11)
    ax2.set_ylabel("Best feasible $L_2$ so far", fontsize=11)
    ax2.set_title("(b) Ladder height sweep — $H=200$ vs $H=300$", fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax2.text(190, 0.083, "0.082", fontsize=8, color="C3", va="bottom", ha="right")
    ax2.text(190, 0.068, "0.067", fontsize=8, color="C4", va="bottom", ha="right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
