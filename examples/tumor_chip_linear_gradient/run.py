"""
Example — inverse design for a linear concentration gradient.

Runs the end-to-end v2 pipeline on a shrunken BO budget so that a fresh
clone can reproduce a reasonable approximation of the paper result in
~15 minutes.  After completion the winner is analysed by Module 3.3 and
a ``design_heuristics.md`` is written into the BO state directory.

Usage:
    python examples/tumor_chip_linear_gradient/run.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ooc_optimizer.config import load_config
from ooc_optimizer.interpretability import analyse_winner
from ooc_optimizer.optimization.orchestrator import run_all_configurations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _merge_overrides(base: dict, override: dict) -> dict:
    for key, value in override.items():
        base[key] = value
    return base


def main() -> None:
    # Start from the top-level default config, then apply the example's
    # overrides.  This keeps the example small and self-documenting.
    config = load_config(REPO / "configs" / "default_config.yaml")
    with open(HERE / "config.yaml", "r") as f:
        overrides = yaml.safe_load(f) or {}
    config = _merge_overrides(config, overrides)

    # Run the single-topology BO.
    results = run_all_configurations(config, parallel=False)
    winner = results.get("winner")
    if winner is None:
        raise SystemExit("No feasible winner found for the linear-gradient target")

    print(
        f"Winner: topology={winner['topology']}, pillar={winner['pillar_config']}, "
        f"H={winner['H']} μm, L2={winner['L2_to_target']:.4f}"
    )

    # Run the interpretability analysis on that winner.
    state_dir = Path(winner["state_dir"])
    summary = analyse_winner(
        state_dir,
        sobol_n_samples=256,
        tolerance_loss_tolerance=0.1,
    )
    print(f"Design heuristics: {summary['heuristic_markdown']}")


if __name__ == "__main__":
    main()
