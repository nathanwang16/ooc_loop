"""
Module 4.1 — 3D CFD validation driver (v2).

Given a saved BO result from Module 3.1 (``evaluations.json`` under a
``bo_<topology>_<pillar>_H<height>`` state dir) and its matching 2D case
directory, re-run the winner in 3D (simpleFoam + scalarTransportFoam) and
emit the three v2 comparison figures plus a JSON summary.

Usage
-----
Single-winner mode::

    python scripts/run_3d_validation.py \\
        --bo-state data/results/bo_opposing_none_H200 \\
        --case-2d data/cases/run_opposing_none_H200_<ts> \\
        --target-profile configs/default_config.yaml \\
        --output data/validation_3d/opposing_linear

Multi-winner mode (auto-pairs primary + secondary winners from a saved
orchestrator summary produced by ``scripts/run_optimization.py``)::

    python scripts/run_3d_validation.py \\
        --orchestrator-summary data/results/optimization_summary.json \\
        --config configs/default_config.yaml \\
        --output data/validation_3d
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ooc_optimizer.config import load_config
from ooc_optimizer.optimization.objectives import build_target_profile
from ooc_optimizer.validation import (
    dump_results,
    plot_all_v2,
    validate_winner_3d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_3d_validation")


def _load_bo_winner(state_dir: Path) -> Dict[str, Any]:
    """Extract the best-feasible record from a BO state directory."""
    evals_path = Path(state_dir) / "evaluations.json"
    if not evals_path.exists():
        raise FileNotFoundError(f"Missing {evals_path}")
    with open(evals_path, "r") as f:
        payload = json.load(f)
    evaluations = payload.get("evaluations") or []
    feasible = [r for r in evaluations if r.get("feasible")]
    if not feasible:
        feasible = evaluations
    if not feasible:
        raise RuntimeError(f"No evaluations recorded in {evals_path}")
    best = min(feasible, key=lambda r: r.get("objective", float("inf")))
    return {
        "params": best["params"],
        "topology": payload["topology"],
        "pillar_config": payload["pillar_config"],
        "H": payload["H"],
        "metrics": best.get("metrics", {}),
        "L2_to_target": best.get("objective"),
        "target_profile": payload.get("target_profile"),
    }


def _run_one(
    *,
    winner: Dict[str, Any],
    config: Dict[str, Any],
    target_spec: Dict[str, Any],
    case_2d: Optional[Path],
    output_dir: Path,
    nz: int,
    z_grading: float,
) -> Dict[str, Any]:
    result = validate_winner_3d(
        winner=winner,
        config=config,
        target_profile_spec=target_spec,
        output_dir=output_dir,
        nz=nz,
        z_grading=z_grading,
    )
    summary = result.to_dict()

    if case_2d is not None and case_2d.exists():
        target = build_target_profile(dict(target_spec))
        mu = float(config["fixed_parameters"]["fluid_viscosity_Pa_s"])
        H_m = float(winner["H"]) * 1e-6
        # Chamber L/W from fixed parameters + winner.
        L_m = float(config["fixed_parameters"]["chamber_length_um"]) * 1e-6
        W_m = float(winner["params"]["W"]) * 1e-6
        figs = plot_all_v2(
            case_2d=case_2d,
            case_3d=result.case_dir,
            target=target,
            L=L_m,
            W=W_m,
            H_m=H_m,
            mu=mu,
            output_dir=output_dir / "figures",
        )
        summary["comparison_plots"] = figs
    else:
        logger.warning("No 2D case provided; skipping 2D-vs-3D figures")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config for paths + mu + diffusivity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nz", type=int, default=25)
    parser.add_argument("--z-grading", type=float, default=1.0)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--bo-state", type=Path, help="Single BO state directory")
    g.add_argument(
        "--orchestrator-summary",
        type=Path,
        help="optimization_summary.json produced by scripts/run_optimization.py",
    )
    parser.add_argument("--case-2d", type=Path, help="Matching 2D case (for comparison plots)")
    parser.add_argument(
        "--target-profile",
        type=Path,
        help="YAML with target_profile block; overrides config / BO default.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    results = []
    if args.bo_state:
        winner = _load_bo_winner(args.bo_state)
        target_spec: Dict[str, Any] = {}
        if args.target_profile is not None:
            import yaml

            with open(args.target_profile, "r") as f:
                target_spec = (yaml.safe_load(f) or {}).get("target_profile", {})
        if not target_spec:
            target_spec = winner.get("target_profile") or config.get("target_profile") or {"kind": "linear_gradient"}

        summary = _run_one(
            winner=winner,
            config=config,
            target_spec=target_spec,
            case_2d=args.case_2d,
            output_dir=args.output,
            nz=args.nz,
            z_grading=args.z_grading,
        )
        results.append(summary)
    else:
        with open(args.orchestrator_summary, "r") as f:
            orch = json.load(f)
        primary_winner = (orch.get("primary") or {}).get("winner")
        if primary_winner is not None:
            target_spec = (
                primary_winner.get("target_profile")
                or config.get("target_profile")
                or {"kind": "linear_gradient"}
            )
            results.append(
                _run_one(
                    winner=primary_winner,
                    config=config,
                    target_spec=target_spec,
                    case_2d=args.case_2d,
                    output_dir=args.output / "primary",
                    nz=args.nz,
                    z_grading=args.z_grading,
                )
            )
        for idx, sec in enumerate(orch.get("secondary", []) or []):
            sw = sec.get("winner")
            if sw is None:
                continue
            target_spec = sec.get("target_profile") or {"kind": "linear_gradient"}
            results.append(
                _run_one(
                    winner=sw,
                    config=config,
                    target_spec=target_spec,
                    case_2d=args.case_2d,
                    output_dir=args.output / f"secondary_{idx}",
                    nz=args.nz,
                    z_grading=args.z_grading,
                )
            )

    out = args.output / "validation_3d_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Summary written to %s", out)


if __name__ == "__main__":
    main()
