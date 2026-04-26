"""
Phase 2 — Sobol-quasirandom scan over (W, Q_total) for the N=8 ladder topology
against the linear-gradient axis=y target.

Same physics path as scripts/diagnostic_ladder_baseline.py: builds the ladder
blockMeshDict, solves simpleFoam + scalarTransportFoam, computes L2 against
axis=y linear gradient.  Loops the scan over a Sobol design in
(W ∈ [1500, 4500] μm, Q_total ∈ [5, 200] μL/min); other parameters are fixed
to match the baseline diagnostic.

This is a stand-in for a full BO campaign — adequate to map the response
surface and confirm whether the topology's L2 floor is materially below the
0.1756 baseline.  If the scan finds a clear minimum, full BO integration
(generalising the production BC writers / orchestrator to support N-inlet
topologies) is justified.

Output:
    examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/results.jsonl
    examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/summary.json
    examples/tumor_chip_linear_gradient/data/diagnostic/ladder_scan/heatmap.png
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ooc_optimizer.geometry.topology_blockmesh import _render
from ooc_optimizer.cfd.scalar import (
    _find_openfoam_prefix,
    extract_concentration_field,
    run_scalar_transport,
)
from ooc_optimizer.optimization.objectives import build_target_profile, l2_to_target

# Reuse the ladder-blockmesh builder from the baseline diagnostic.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from diagnostic_ladder_baseline import (  # noqa: E402
    build_ladder_blockmesh,
    write_u_field,
    write_p_field,
    write_t_field,
    run_blockmesh,
    run_simplefoam,
    run_postprocess,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("diagnostic_ladder_scan")


N_INLETS = 8
L_MM = 10.0
H_UM = 200.0
DZ_MM = H_UM / 1000.0
DIFFUSIVITY = 1.0e-10
BASE_NX = 200
NY_PER_MM = 25

W_BOUNDS_UM = (1500.0, 4500.0)
Q_BOUNDS_UL_MIN = (5.0, 200.0)


def evaluate(W_um: float, Q_total_uL_min: float, scan_root: Path, idx: int) -> dict:
    W_mm = W_um / 1000.0
    bm_text, inlet_names, _ = build_ladder_blockmesh(
        N=N_INLETS, W_mm=W_mm, L_mm=L_MM, dz_mm=DZ_MM, base_nx=BASE_NX, ny_per_mm=NY_PER_MM,
    )
    C_values = [k / (N_INLETS - 1) for k in range(N_INLETS)]
    Q_m3s = Q_total_uL_min * 1e-9 / 60.0
    inlet_area_m2 = (W_mm / N_INLETS * 1e-3) * (DZ_MM * 1e-3)
    U_in = (Q_m3s / N_INLETS) / inlet_area_m2

    case_dir = scan_root / f"case_{idx:03d}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    template_dir = REPO_ROOT / "ooc_optimizer" / "cfd" / "template"
    shutil.copytree(template_dir, case_dir)
    (case_dir / "system" / "blockMeshDict").write_text(bm_text)
    write_u_field(case_dir, inlet_names, U_in)
    write_p_field(case_dir, inlet_names)
    write_t_field(case_dir, inlet_names, C_values)

    t0 = time.time()
    try:
        run_blockmesh(case_dir)
        run_simplefoam(case_dir, timeout=300)
        run_postprocess(case_dir, "writeCellCentres")
        scalar_patches = {
            **{name: f"fixedValue:{c}" for name, c in zip(inlet_names, C_values)},
            "outlet": "zeroGradient",
            "walls": "zeroGradient",
            "floor": "zeroGradient",
            "frontAndBack": "empty",
        }
        scalar_result = run_scalar_transport(
            case_dir=case_dir,
            diffusivity_m2_s=DIFFUSIVITY,
            c_drug=1.0, c_medium=0.0,
            patches=scalar_patches,
        )
        centres, C = extract_concentration_field(case_dir)
        L_m = L_MM * 1e-3
        W_m = W_mm * 1e-3
        target = build_target_profile({"kind": "linear_gradient", "axis": "y", "c_high": 1.0, "c_low": 0.0})
        C_target = target.evaluate(centres[:, 0], centres[:, 1], L=L_m, W=W_m)
        L2 = float(l2_to_target(C, C_target))
        wall = time.time() - t0

        # Save space — keep only the latest time directory of the case_NNN dir.
        # (Skip cleanup for simplicity; ~50MB per case is OK on a 200GB disk.)

        return {
            "idx": idx, "W_um": W_um, "Q_total_uL_min": Q_total_uL_min,
            "L2_to_target_axis_y": L2,
            "C_mean": float(np.mean(C)), "C_std": float(np.std(C)),
            "C_min": float(np.min(C)), "C_max": float(np.max(C)),
            "scalar_converged": bool(scalar_result.converged),
            "U_in_m_s": U_in, "wall_time_s": wall,
            "case_dir": str(case_dir), "ok": True,
        }
    except Exception as exc:
        wall = time.time() - t0
        log.error("eval %d failed (W=%.0f Q=%.2f): %s", idx, W_um, Q_total_uL_min, exc)
        return {
            "idx": idx, "W_um": W_um, "Q_total_uL_min": Q_total_uL_min,
            "L2_to_target_axis_y": float("nan"),
            "scalar_converged": False, "wall_time_s": wall,
            "error": str(exc), "ok": False,
        }


def main() -> int:
    n_evals = int(os.environ.get("N_EVALS", "32"))
    seed = int(os.environ.get("SEED", "42"))

    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    unit = sampler.random(n=n_evals)
    Ws = W_BOUNDS_UM[0] + unit[:, 0] * (W_BOUNDS_UM[1] - W_BOUNDS_UM[0])
    Qs = Q_BOUNDS_UL_MIN[0] + unit[:, 1] * (Q_BOUNDS_UL_MIN[1] - Q_BOUNDS_UL_MIN[0])

    scan_root = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "diagnostic" / "ladder_scan"
    scan_root.mkdir(parents=True, exist_ok=True)
    results_path = scan_root / "results.jsonl"
    summary_path = scan_root / "summary.json"
    heatmap_path = scan_root / "heatmap.png"

    log.info("Sobol scan: %d evals over W ∈ %s μm, Q ∈ %s μL/min", n_evals, W_BOUNDS_UM, Q_BOUNDS_UL_MIN)
    rows: list[dict] = []
    with open(results_path, "w") as fh:
        for i in range(n_evals):
            r = evaluate(float(Ws[i]), float(Qs[i]), scan_root, i)
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            log.info("  [%2d/%d] W=%.0f Q=%.2f  L2=%.4f  C_mean=%.3f  wall=%.1fs",
                     i + 1, n_evals, r["W_um"], r["Q_total_uL_min"],
                     r.get("L2_to_target_axis_y", float("nan")),
                     r.get("C_mean", float("nan")), r["wall_time_s"])

    feasible = [r for r in rows if r["ok"] and r["L2_to_target_axis_y"] == r["L2_to_target_axis_y"]]
    if not feasible:
        log.error("No feasible evaluations.")
        return 1
    best = min(feasible, key=lambda r: r["L2_to_target_axis_y"])

    summary = {
        "n_evals": n_evals,
        "n_feasible": len(feasible),
        "best": best,
        "L2_baseline_uniform_W3000": 0.1756,   # from diagnostic_ladder_baseline.py
        "L2_old_BO_winner_opposing": 0.6343,
        "W_bounds_um": W_BOUNDS_UM, "Q_bounds_uL_min": Q_BOUNDS_UL_MIN,
        "N_inlets": N_INLETS, "L_mm": L_MM, "H_um": H_UM,
        "diffusivity_m2_s": DIFFUSIVITY,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    Ws_f = np.array([r["W_um"] for r in feasible])
    Qs_f = np.array([r["Q_total_uL_min"] for r in feasible])
    L2s = np.array([r["L2_to_target_axis_y"] for r in feasible])
    sc = ax.scatter(Ws_f, Qs_f, c=L2s, s=80, cmap="viridis_r", edgecolor="black", linewidth=0.4)
    ax.scatter(best["W_um"], best["Q_total_uL_min"], s=300, marker="*", color="red",
               edgecolor="black", linewidth=1.0, label=f"best L2={best['L2_to_target_axis_y']:.4f}")
    plt.colorbar(sc, label="L2 vs linear y-gradient")
    ax.set_xlabel("W (chamber width) [μm]")
    ax.set_ylabel("Q_total [μL/min]")
    ax.set_title(f"Ladder N=8 — {len(feasible)} Sobol evals over (W, Q_total)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=140)

    print()
    print("=== Phase 2 — Sobol scan over ladder (N=8) topology ===")
    print(f"  evals:          {n_evals} ({len(feasible)} feasible)")
    print(f"  best L2:        {best['L2_to_target_axis_y']:.4f}")
    print(f"  best W:         {best['W_um']:.0f} μm")
    print(f"  best Q_total:   {best['Q_total_uL_min']:.2f} μL/min")
    print(f"  baseline (W=3000, Q=50): 0.1756")
    print(f"  old opposing BO winner:  0.6343")
    improve = 100.0 * (0.1756 - best["L2_to_target_axis_y"]) / 0.1756
    print(f"  improvement vs baseline: {improve:+.1f}%")
    print(f"\nArtifacts: {results_path.relative_to(REPO_ROOT)}")
    print(f"           {summary_path.relative_to(REPO_ROOT)}")
    print(f"           {heatmap_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
