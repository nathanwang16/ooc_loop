"""
Run the WSS uniformity premise test sweep and generate a report bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ooc_optimizer.analysis.wss_contours import plot_side_by_side
from ooc_optimizer.cfd.inlet_premise_runner import run_grid
from ooc_optimizer.config import load_config
from ooc_optimizer.validation.cfd_3d import compare_2d_vs_3d_matched, run_3d_matched_rectangle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_inlet_premise_test")


def parse_grid(grid_text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for chunk in grid_text.split(";"):
        key, values = chunk.split("=", maxsplit=1)
        key = key.strip()
        nums = [float(v.strip()) for v in values.split(",") if v.strip()]
        if not nums:
            raise ValueError(f"No values provided for grid key '{key}'")
        out[key] = nums
    for required in ("W", "H", "Q"):
        if required not in out:
            raise ValueError(f"Grid must include {required}")
    return out


def verdict_for_pair(matched: dict[str, Any], mismatched: dict[str, Any], has_3d_flat_profile: bool) -> str:
    delta_dev = mismatched["cv_developed"] - matched["cv_developed"]
    delta_core = mismatched["cv_core"] - matched["cv_core"]
    if delta_dev < 0.05 and abs(delta_core) < 0.05:
        return "A_jet_only_entrance_artifact"
    if delta_dev > 0.10:
        return "B_jet_penetrates_premise_intact_regime"
    if (
        matched["cv_developed"] > 0.30
        and matched["cv_core"] < 0.10
        and has_3d_flat_profile
    ):
        return "C_2d_methodology_bound"
    return "D_inconclusive"


def _index_results(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (int(round(row["W_um"])), int(round(row["H_um"])), int(round(row["Q_ul_min"])))
        indexed[key] = row
    return indexed


def _plot_centerline_pair(
    matched: dict[str, Any],
    mismatched: dict[str, Any],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.6))
    plt.plot(matched["centerline_tau_of_x"]["x_m"], matched["centerline_tau_of_x"]["tau_pa"], label="matched")
    plt.plot(mismatched["centerline_tau_of_x"]["x_m"], mismatched["centerline_tau_of_x"]["tau_pa"], label="mismatched")
    plt.xlabel("x (m)")
    plt.ylabel("centerline tau (Pa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def _plot_cv_y_of_x_pair(
    matched: dict[str, Any],
    mismatched: dict[str, Any],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.6))
    plt.plot(matched["cv_y_of_x"]["x_m"], matched["cv_y_of_x"]["cv"], label="matched")
    plt.plot(mismatched["cv_y_of_x"]["x_m"], mismatched["cv_y_of_x"]["cv"], label="mismatched")
    plt.xlabel("x (m)")
    plt.ylabel("CV_y(tau)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def _is_3d_profile_flat(results_3d: dict[str, Any], max_cv: float = 0.15) -> bool:
    # Proxy criterion: if depth-averaged 3D core CV proxy is low, treat profile as flat.
    cv_core = results_3d.get("cv_core_3d_proxy", np.nan)
    return bool(np.isfinite(cv_core) and cv_core < max_cv)


def _write_summary_csv(
    output_dir: Path,
    matched_rows: list[dict[str, Any]],
    mismatched_rows: list[dict[str, Any]],
    has_3d_flat_profile: bool,
) -> Path:
    path = output_dir / "summary.csv"
    matched_idx = _index_results(matched_rows)
    mismatched_idx = _index_results(mismatched_rows)
    keys = sorted(set(matched_idx) & set(mismatched_idx))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "W_um",
                "H_um",
                "Q_ul_min",
                "Re_matched",
                "Re_mismatched",
                "cv_global_matched",
                "cv_global_mismatched",
                "cv_developed_matched",
                "cv_developed_mismatched",
                "cv_core_matched",
                "cv_core_mismatched",
                "cv_core_central_matched",
                "cv_core_central_mismatched",
                "entrance_len_m_matched",
                "entrance_len_m_mismatched",
                "verdict",
            ]
        )
        for key in keys:
            m = matched_idx[key]
            mm = mismatched_idx[key]
            verdict = verdict_for_pair(matched=m, mismatched=mm, has_3d_flat_profile=has_3d_flat_profile)
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    m["Re"],
                    mm["Re"],
                    m["cv_global"],
                    mm["cv_global"],
                    m["cv_developed"],
                    mm["cv_developed"],
                    m["cv_core"],
                    mm["cv_core"],
                    m["cv_core_central"],
                    mm["cv_core_central"],
                    m["entrance_length_estimate_m"],
                    mm["entrance_length_estimate_m"],
                    verdict,
                ]
            )
    return path


def _write_report(
    output_dir: Path,
    grid: dict[str, list[float]],
    matched_rows: list[dict[str, Any]],
    mismatched_rows: list[dict[str, Any]],
    summary_csv: Path,
    contour_path: Path,
    centerline_path: Path,
    cv_y_path: Path,
    compare_2d_3d: dict[str, Any] | None,
    results_3d: dict[str, Any] | None,
) -> Path:
    matched_idx = _index_results(matched_rows)
    mismatched_idx = _index_results(mismatched_rows)
    keys = sorted(set(matched_idx) & set(mismatched_idx))
    has_3d_flat_profile = _is_3d_profile_flat(results_3d or {})
    lines = []
    lines.append("# WSS Uniformity Premise Test Report\n\n")
    lines.append(f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("## Test Grid\n\n")
    lines.append(
        f"- W (um): {', '.join(str(int(v)) for v in grid['W'])}\n"
        f"- H (um): {', '.join(str(int(v)) for v in grid['H'])}\n"
        f"- Q (uL/min): {', '.join(str(int(v)) for v in grid['Q'])}\n"
    )
    lines.append("\n## Per-case Verdicts\n\n")
    lines.append("| W | H | Q | CV_dev matched | CV_dev mismatched | CV_core matched | CV_core mismatched | Verdict |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---|\n")
    for key in keys:
        m = matched_idx[key]
        mm = mismatched_idx[key]
        verdict = verdict_for_pair(matched=m, mismatched=mm, has_3d_flat_profile=has_3d_flat_profile)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {m['cv_developed']:.4f} | {mm['cv_developed']:.4f} | "
            f"{m['cv_core']:.4f} | {mm['cv_core']:.4f} | {verdict} |\n"
        )

    lines.append("\n## Artifacts\n\n")
    lines.append(f"- Summary CSV: `{summary_csv}`\n")
    lines.append(f"- Side-by-side contour: `{contour_path}`\n")
    lines.append(f"- Centerline tau(x): `{centerline_path}`\n")
    lines.append(f"- CV_y(x): `{cv_y_path}`\n")
    if compare_2d_3d is not None:
        lines.append(f"- 2D vs 3D y-profile plot: `{compare_2d_3d['profile_plot']}`\n")
        lines.append(f"- 2D vs 3D scatter/Bland-Altman: `{compare_2d_3d['scatter_bland_altman_plot']}`\n")
    if results_3d is not None:
        lines.append(f"- 3D metrics JSON: `{Path(results_3d['case_dir']) / 'metrics_3d.json'}`\n")

    lines.append("\n## ParaView Instructions\n\n")
    lines.append("### 2D Cases\n\n")
    lines.append("1. Open `<case_dir>/case.foam` with `OpenFOAMReader`.\n")
    lines.append("2. Color by `U` magnitude to inspect jet structure.\n")
    lines.append("3. Create `tau_floor` with Calculator: `6.0 * 0.001 * mag(U) / <H_m>`.\n")
    lines.append("4. Clip to developed region: x in `[L_stub + 1 mm, L_stub + L_chamber - 1 mm]`.\n")
    lines.append("5. Clip sidewalls for core region: y in `[2H, W - 2H]`.\n")
    lines.append("6. For CV_y(x) checks: Slice at fixed x and use Plot On Sorted Lines for tau(y).\n\n")
    lines.append("### 3D Sanity Case\n\n")
    lines.append("1. Open `<3d_case>/case.foam` with `OpenFOAMReader`.\n")
    lines.append("2. Color by `U` and inspect floor/sidewall effects.\n")
    lines.append("3. Load `wallShearStress` (from `postProcess -func wallShearStress(U)`) and color by magnitude on the floor patch.\n")
    lines.append("4. For depth-averaged comparison, slice at several z values or use Integrate Variables along z, then compare against 2D maps.\n")
    lines.append("5. Open 2D and 3D side-by-side with linked cameras for direct visual contrast.\n")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched vs mismatched WSS premise test.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config yaml.")
    parser.add_argument("--mode", choices=["matched", "mismatched", "both"], default="both")
    parser.add_argument(
        "--grid",
        default="W=1500,2500,3000;H=200,300;Q=50,200",
        help="Grid spec string, e.g. W=1500,2500,3000;H=200,300;Q=50,200",
    )
    parser.add_argument("--residual-tol", type=float, default=1e-6)
    parser.add_argument("--run-3d-sanity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_root = output_dir / "cases"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    grid = parse_grid(args.grid)
    mu = float(cfg["fixed_parameters"]["fluid_viscosity_Pa_s"])
    rho = float(cfg["fixed_parameters"]["fluid_density_kg_m3"])

    matched_rows: list[dict[str, Any]] = []
    mismatched_rows: list[dict[str, Any]] = []

    if args.mode in {"matched", "both"}:
        matched_rows = run_grid(
            grid=grid,
            inlet_mode="matched",
            work_dir=cases_root / "matched",
            mu=mu,
            rho=rho,
            residual_tol=args.residual_tol,
            template_dir=Path(cfg["paths"]["template_case"]),
        )
    if args.mode in {"mismatched", "both"}:
        mismatched_rows = run_grid(
            grid=grid,
            inlet_mode="mismatched",
            work_dir=cases_root / "mismatched",
            mu=mu,
            rho=rho,
            residual_tol=args.residual_tol,
            template_dir=Path(cfg["paths"]["template_case"]),
        )

    (output_dir / "matched_results.json").write_text(json.dumps(matched_rows, indent=2), encoding="utf-8")
    (output_dir / "mismatched_results.json").write_text(json.dumps(mismatched_rows, indent=2), encoding="utf-8")

    # Representative comparison case
    key = (2500, 200, 50)
    matched_idx = _index_results(matched_rows) if matched_rows else {}
    mismatched_idx = _index_results(mismatched_rows) if mismatched_rows else {}
    if key not in matched_idx or key not in mismatched_idx:
        raise RuntimeError(
            f"Representative case {key} missing. Check --grid and --mode."
        )
    matched_rep = matched_idx[key]
    mismatched_rep = mismatched_idx[key]

    contour_path = figures_dir / "matched_vs_mismatched_contour_W2500_H200_Q50.png"
    plot_side_by_side(
        baseline_case=Path(matched_rep["case_dir"]),
        optimized_case=Path(mismatched_rep["case_dir"]),
        H=200e-6,
        mu=mu,
        output_path=contour_path,
    )
    centerline_path = _plot_centerline_pair(
        matched=matched_rep,
        mismatched=mismatched_rep,
        out_path=figures_dir / "centerline_tau_x_W2500_H200_Q50.png",
    )
    cv_y_path = _plot_cv_y_of_x_pair(
        matched=matched_rep,
        mismatched=mismatched_rep,
        out_path=figures_dir / "cv_y_of_x_W2500_H200_Q50.png",
    )

    results_3d: dict[str, Any] | None = None
    compare_2d_3d: dict[str, Any] | None = None
    if args.run_3d_sanity:
        results_3d = run_3d_matched_rectangle(
            W_um=2500.0,
            H_um=200.0,
            L_um=float(cfg["fixed_parameters"]["chamber_length_um"]),
            Q_ul_min=50.0,
            mu=mu,
            rho=rho,
            work_dir=cases_root / "sanity_3d",
            residual_tol=args.residual_tol,
        )
        compare_2d_3d = compare_2d_vs_3d_matched(
            case_2d_dir=Path(matched_rep["case_dir"]),
            case_3d_dir=Path(results_3d["case_dir"]),
            output_dir=figures_dir,
        )
        (output_dir / "compare_2d_vs_3d.json").write_text(json.dumps(compare_2d_3d, indent=2), encoding="utf-8")

    summary_csv = _write_summary_csv(
        output_dir=output_dir,
        matched_rows=matched_rows,
        mismatched_rows=mismatched_rows,
        has_3d_flat_profile=_is_3d_profile_flat(results_3d or {}),
    )

    report_path = _write_report(
        output_dir=output_dir,
        grid=grid,
        matched_rows=matched_rows,
        mismatched_rows=mismatched_rows,
        summary_csv=summary_csv,
        contour_path=contour_path,
        centerline_path=centerline_path,
        cv_y_path=cv_y_path,
        compare_2d_3d=compare_2d_3d,
        results_3d=results_3d,
    )
    logger.info("Wrote report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
