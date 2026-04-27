"""Re-render the Phase-2 ladder-scan (W, Q_total) heatmap with larger fonts.

Reads `examples/.../data/diagnostic/ladder_scan/results.jsonl` (no CFD re-run)
and overwrites `heatmap.png` in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "diagnostic" / "ladder_scan"


def main() -> int:
    rows = [json.loads(l) for l in (SCAN / "results.jsonl").read_text().splitlines() if l.strip()]
    feasible = [r for r in rows if r.get("feasible", True)]
    Ws = np.array([r["W_um"] for r in feasible])
    Qs = np.array([r["Q_total_uL_min"] for r in feasible])
    L2s = np.array([r["L2_to_target_axis_y"] for r in feasible])
    best = min(feasible, key=lambda r: r["L2_to_target_axis_y"])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sc = ax.scatter(Ws, Qs, c=L2s, s=160, cmap="viridis_r",
                    edgecolor="black", linewidth=0.6)
    ax.scatter(best["W_um"], best["Q_total_uL_min"],
               s=520, marker="*", color="red", edgecolor="black", linewidth=1.4,
               label=f"best L2 = {best['L2_to_target_axis_y']:.4f}")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("$L_2$ vs linear y-gradient", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    ax.set_xlabel("W  (chamber width, μm)", fontsize=15, labelpad=8)
    ax.set_ylabel("Q$_{total}$  (μL/min)", fontsize=15, labelpad=8)
    ax.set_title(f"Ladder N=8 — {len(feasible)} Sobol evals over (W, Q$_{{total}}$)",
                 fontsize=16, pad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper right", fontsize=14)
    fig.tight_layout()

    out = SCAN / "heatmap.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
