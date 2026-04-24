"""Finalize premise-test report from an existing output directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_inlet_premise_test import _is_3d_profile_flat, _write_report, _write_summary_csv
from ooc_optimizer.cfd.foam_parser import find_latest_time, read_cell_centres, read_vector_field
from ooc_optimizer.validation.cfd_3d import (
    _depth_average_velocity,
    _estimate_floor_shear_from_near_wall_cells,
    _parse_floor_wall_shear_magnitudes,
    _safe_cv,
    compare_2d_vs_3d_matched,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", default="W=1500,2500,3000;H=200,300;Q=50,200")
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--h-um", type=float, default=200.0)
    parser.add_argument("--rep-key", default="2500,200,50")
    args = parser.parse_args()

    out = args.output_dir
    matched_rows = json.loads((out / "matched_results.json").read_text(encoding="utf-8"))
    mismatched_rows = json.loads((out / "mismatched_results.json").read_text(encoding="utf-8"))

    rep_w, rep_h, rep_q = [int(x.strip()) for x in args.rep_key.split(",")]
    matched_idx = {
        (int(round(r["W_um"])), int(round(r["H_um"])), int(round(r["Q_ul_min"]))): r
        for r in matched_rows
    }
    rep_case_2d = Path(matched_idx[(rep_w, rep_h, rep_q)]["case_dir"])
    case3d = out / "cases" / "sanity_3d" / f"matched_3d_W{rep_w}_H{rep_h}_Q{rep_q}"

    latest = find_latest_time(case3d)
    if latest is None:
        raise FileNotFoundError(f"No latest time found in {case3d}")

    C = read_cell_centres(case3d)
    U = read_vector_field(latest / "U")
    U_bar = _depth_average_velocity(C=C, U=U, x_round_decimals=9, y_round_decimals=9)
    H_m = args.h_um * 1e-6
    tau_proxy = (6.0 * args.mu * np.linalg.norm(U_bar[:, 2:4], axis=1)) / H_m

    wss_file = latest / "wallShearStress"
    floor = _parse_floor_wall_shear_magnitudes(wss_file)
    if floor.size == 0:
        floor = _estimate_floor_shear_from_near_wall_cells(C=C, U=U, mu=args.mu)

    L_m = 0.01
    W_m = rep_w * 1e-6
    mask_ch = (U_bar[:, 0] >= 0) & (U_bar[:, 0] <= L_m)
    mask_dev = mask_ch & (U_bar[:, 0] >= 1e-3) & (U_bar[:, 0] <= L_m - 1e-3)
    side = 2 * H_m
    mask_core = mask_dev & (U_bar[:, 1] >= side) & (U_bar[:, 1] <= W_m - side)

    results_3d = {
        "case_dir": str(case3d),
        "cv_global_3d_resolved": _safe_cv(floor),
        "cv_global_3d_proxy": _safe_cv(tau_proxy[mask_ch]),
        "cv_developed_3d_proxy": _safe_cv(tau_proxy[mask_dev]),
        "cv_core_3d_proxy": _safe_cv(tau_proxy[mask_core]),
        "tau_floor_3d_mean": float(np.mean(floor)),
        "tau_floor_3d_min": float(np.min(floor)),
        "tau_floor_3d_max": float(np.max(floor)),
        "tau_floor_2d_proxy_mean": float(np.mean(tau_proxy[mask_ch])),
        "wall_shear_direct_available": bool(
            wss_file.exists() and _parse_floor_wall_shear_magnitudes(wss_file).size > 0
        ),
    }
    (case3d / "metrics_3d.json").write_text(json.dumps(results_3d, indent=2), encoding="utf-8")

    compare = compare_2d_vs_3d_matched(
        case_2d_dir=rep_case_2d,
        case_3d_dir=case3d,
        output_dir=out / "figures",
    )
    (out / "compare_2d_vs_3d.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")

    summary_csv = _write_summary_csv(
        output_dir=out,
        matched_rows=matched_rows,
        mismatched_rows=mismatched_rows,
        has_3d_flat_profile=_is_3d_profile_flat(results_3d),
    )

    grid = {}
    for part in args.grid.split(";"):
        key, value = part.split("=")
        grid[key] = [float(v.strip()) for v in value.split(",")]

    report = _write_report(
        output_dir=out,
        grid=grid,
        matched_rows=matched_rows,
        mismatched_rows=mismatched_rows,
        summary_csv=summary_csv,
        contour_path=out / "figures" / "matched_vs_mismatched_contour_W2500_H200_Q50.png",
        centerline_path=out / "figures" / "centerline_tau_x_W2500_H200_Q50.png",
        cv_y_path=out / "figures" / "cv_y_of_x_W2500_H200_Q50.png",
        compare_2d_3d=compare,
        results_3d=results_3d,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
