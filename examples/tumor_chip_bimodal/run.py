"""Example — inverse design for a bimodal (rim + rim) concentration profile."""

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


def main() -> None:
    config = load_config(REPO / "configs" / "default_config.yaml")
    with open(HERE / "config.yaml", "r") as f:
        overrides = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        config[k] = v

    results = run_all_configurations(config, parallel=False)
    winner = results.get("winner")
    if winner is None:
        raise SystemExit("No feasible winner found for the bimodal target")
    print(
        f"Winner: topology={winner['topology']}, pillar={winner['pillar_config']}, "
        f"H={winner['H']} μm, L2={winner['L2_to_target']:.4f}"
    )

    summary = analyse_winner(
        Path(winner["state_dir"]),
        sobol_n_samples=256,
        tolerance_loss_tolerance=0.1,
    )
    print(f"Design heuristics: {summary['heuristic_markdown']}")


if __name__ == "__main__":
    main()
