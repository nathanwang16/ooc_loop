"""
Module 3.1 — Run the full Bayesian optimization campaign (v2).

Executes BO for all 24 discrete configurations (4 pillars × 2 heights × 3
inlet topologies) on the primary target profile, then re-runs the winning
topology on the remaining two target profiles.  Reports the overall winner
and the per-target best.

Usage:
    python scripts/run_optimization.py --config configs/default_config.yaml
    python scripts/run_optimization.py --config configs/default_config.yaml --parallel
    python scripts/run_optimization.py --config configs/default_config.yaml \
        --single-target                # sweep 24 configs on primary target only

Pass ``--override configs/examples/bimodal.yaml`` to override the primary
target profile in the default config with the target defined in a sibling
example YAML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from ooc_optimizer.config import load_config
from ooc_optimizer.optimization.orchestrator import (
    run_all_configurations,
    run_multi_target_workflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _apply_override(config: dict, override_path: Path) -> dict:
    with open(override_path, "r") as f:
        ov = yaml.safe_load(f) or {}
    for key, value in ov.items():
        config[key] = value
    logger.info("Applied override from %s (keys: %s)", override_path, list(ov))
    return config


def main():
    parser = argparse.ArgumentParser(description="Run BO optimization campaign (v2)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--single-target",
        action="store_true",
        help="Sweep 24 configurations on the primary target only; skip the "
        "winner-topology secondary sweeps.",
    )
    parser.add_argument(
        "--override",
        type=Path,
        help="YAML file whose top-level keys override the main config in-place.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/results/optimization_summary.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = _apply_override(config, args.override)

    if args.single_target:
        results = run_all_configurations(config, parallel=args.parallel)
        winner = results.get("winner")
    else:
        results = run_multi_target_workflow(config, parallel=args.parallel)
        winner = (results.get("primary") or {}).get("winner")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(_strip_nonserialisable(results), f, indent=2, default=str)
    logger.info("Summary written to %s", args.summary_out)

    if winner:
        logger.info(
            "Winner: %s (topology=%s, pillar=%s, H=%s) L2=%.4f",
            winner["config_name"],
            winner.get("topology"),
            winner.get("pillar_config"),
            winner.get("H"),
            winner.get("L2_to_target", float("nan")),
        )
    else:
        logger.warning("No feasible solution found")


def _strip_nonserialisable(obj):
    """Drop embedded pydantic / tensor values that json.dump can't round-trip."""
    if isinstance(obj, dict):
        return {k: _strip_nonserialisable(v) for k, v in obj.items() if k != "evaluations" or True}
    if isinstance(obj, list):
        return [_strip_nonserialisable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


if __name__ == "__main__":
    main()
