"""
Module 3.1 — Run the full Bayesian optimization campaign.

Executes BO for all 8 discrete configurations (4 pillar × 2 heights)
and reports the overall winner.

Usage:
    python scripts/run_optimization.py --config configs/default_config.yaml
    python scripts/run_optimization.py --config configs/default_config.yaml --parallel
"""

import argparse
import logging
import sys
from pathlib import Path

from ooc_optimizer.config import load_config
from ooc_optimizer.optimization import run_all_configurations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run BO optimization campaign")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run configurations in parallel"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_all_configurations(config, parallel=args.parallel)

    winner = results.get("winner")
    if winner:
        logger.info("Winner: %s, CV(τ) = %.4f", winner["config_name"], winner["cv_tau"])
    else:
        logger.warning("No feasible solution found across all configurations")


if __name__ == "__main__":
    main()
