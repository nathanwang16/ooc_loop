"""
One-shot STL + mesh + simpleFoam run for manual inspection.

Usage (OpenFOAM must be on PATH, e.g. after sourcing etc/bashrc):

    source /Volumes/OpenFOAM-v2406/etc/bashrc
    python scripts/run_single_verification.py
    python scripts/run_single_verification.py --complex

``--complex`` uses the heaviest valid discrete layout (3x6 pillars, H=300 um),
wide chamber, large pillars, and a non-trivial inlet taper.

Writes outputs under data/manual_verification/ or data/manual_verification_complex/,
a ParaView marker ``case.foam`` in the OpenFOAM case (see tip.md), and a manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ooc_optimizer.config import load_config
from ooc_optimizer.geometry import generate_chip
from ooc_optimizer.optimization.bo_loop import BORunner
from ooc_optimizer.cfd.solver import evaluate_cfd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_single_verification")


def _complex_params() -> tuple[dict, str, float]:
    """Max discrete complexity: 3x6 pillars, H=300 um, valid bounds and constraints."""
    params = {
        "W": 3000.0,
        "d_p": 400.0,
        "s_p": 650.0,
        "theta": 45.0,
        "Q": 150.0,
    }
    return params, "3x6", 300.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Single STL + mesh + CFD verification run.")
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Use 3x6 pillars, H=300 um, and aggressive valid continuous parameters.",
    )
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "configs" / "default_config.yaml")
    base_name = "manual_verification_complex" if args.complex else "manual_verification"
    base = REPO_ROOT / "data" / base_name
    stl_dir = base / "stl"
    cases_dir = base / "cfd_cases"
    stl_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["stl_output_dir"] = str(stl_dir)
    cfg["paths"]["case_output_dir"] = str(cases_dir)

    if args.complex:
        params, pillar, H_um = _complex_params()
        logger.info(
            "Complex preset: pillar_config=%s H=%.0f um W=%.0f d_p=%.0f s_p=%.0f theta=%.1f Q=%.1f",
            pillar,
            H_um,
            params["W"],
            params["d_p"],
            params["s_p"],
            params["theta"],
            params["Q"],
        )
    else:
        baseline = cfg.get("baseline") or {}
        pillar = str(baseline.get("pillar_config", "none"))
        H_um = float(baseline.get("H", 200.0))
        params = {
            "W": float(baseline.get("W", 1500.0)),
            "theta": float(baseline.get("theta", 90.0)),
            "Q": float(baseline.get("Q", 50.0)),
        }
        if pillar.lower() != "none":
            params["d_p"] = float(baseline.get("d_p", 200.0))
            params["s_p"] = float(baseline.get("s_p", 400.0))

    logger.info("Generating chip STLs (fluid + mold)…")
    fluid_stl, mold_stl = generate_chip(params, pillar, H_um, stl_dir)

    cfd_cfg = BORunner._build_cfd_config(cfg)
    logger.info("Running evaluate_cfd (mesh + simpleFoam + metrics)…")
    metrics = evaluate_cfd(params, pillar, H_um, cfd_cfg)

    case_dir = Path(metrics.get("case_dir") or "")
    foam_marker = case_dir / "case.foam"
    if case_dir.is_dir():
        foam_marker.write_text("", encoding="utf-8")
        logger.info("ParaView marker: %s (open this file; select OpenFOAM reader)", foam_marker)
    manifest = base / "VERIFICATION_MANIFEST.md"
    metrics_json = base / "metrics.json"

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    latest_time = None
    if case_dir and case_dir.is_dir():
        numeric_times = [
            p for p in case_dir.iterdir()
            if p.is_dir() and p.name.isdigit() and int(p.name) > 0
        ]
        if numeric_times:
            latest_time = max(numeric_times, key=lambda x: int(x.name)).name

    pillar_stl_row = ""
    if pillar.lower() != "none":
        pillar_stl_row = (
            f"| Pillar obstacle STL | `{case_dir / 'constant' / 'triSurface' / 'pillars.stl'}` |\n"
        )
    else:
        pillar_stl_row = (
            "| Pillar obstacle STL | *(not used — `pillar_config=none`; `triSurface/` empty)* |\n"
        )

    run_label = "complex (3x6, H=300)" if args.complex else "baseline from config"
    lines = [
        "# Manual verification run — STL, mesh, CFD\n",
        f"\nGenerated: {now}\n",
        f"\n**Preset:** {run_label}\n",
    ]
    if args.complex:
        lines.append(
            "\n**Note:** Complex pillar cases often keep a `snappyHexMesh` mesh even when "
            "`checkMesh` reports issues (`mesh_ok` may be false); see `metrics.json` and `tip.md`. "
            "Fields are still written for visualization.\n",
        )
    lines += [
        "\nOpenFOAM used from your environment (`which simpleFoam` at run time).\n",
        "\n## Geometry (CadQuery export)\n",
        f"\n| File | Path |\n|------|------|\n",
        f"| Fluid domain STL | `{fluid_stl.resolve()}` |\n",
        f"| Mold STL | `{mold_stl.resolve()}` |\n",
        "\nInspect in MeshLab, Blender, or ParaView (STL).\n",
        "\n## CFD case (mesh + fields)\n",
        f"\n| Item | Path |\n|------|------|\n",
        f"| Case root | `{case_dir.resolve() if case_dir else '(unknown)'}` |\n",
        f"| PolyMesh | `{case_dir / 'constant' / 'polyMesh'}` |\n",
        pillar_stl_row,
        f"| blockMesh log | `{case_dir / 'logs' / 'blockMesh.log'}` |\n",
        f"| snappyHexMesh log | `{case_dir / 'logs' / 'snappyHexMesh.log'}` |\n",
        f"| checkMesh log | `{case_dir / 'logs' / 'checkMesh.log'}` |\n",
        f"| simpleFoam log | `{case_dir / 'simpleFoam.log'}` |\n",
    ]
    if latest_time:
        lines.append(
            f"| Latest time (fields) | `{case_dir / latest_time}` — contains `U`, `p` |\n",
        )
    lines += [
        "\n**ParaView:** do not rely on opening the case directory with *All Files* (wrong reader). "
        f"Open **`{foam_marker.resolve()}`** and choose **OpenFOAMReader** / **POpenFOAMReader** "
        "(empty marker file — see `tip.md`). Select the latest time; color by `U` or `p`.\n",
    ]
    lines += [
        "\n## Metrics (Python extraction)\n",
        f"\nFull dict: `{metrics_json.resolve()}`\n",
        "\n```json\n",
        json.dumps(metrics, indent=2),
        "\n```\n",
    ]
    manifest.write_text("".join(lines), encoding="utf-8")

    logger.info("Wrote manifest: %s", manifest)
    logger.info("Case directory: %s", case_dir)
    if metrics.get("cv_tau", 999) >= 999 and not metrics.get("converged"):
        logger.warning("Run may have failed (penalty-like metrics); check logs in case dir.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
