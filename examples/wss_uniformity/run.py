"""
Retained v1 example — WSS-uniformity optimisation.

Demonstrates that the same engine solves a completely different scalar
inverse-design problem (floor wall-shear-stress uniformity) without code
changes.  The primary objective drops out of the v2 metrics dict as
``cv_tau`` (retained from v1) and the v2 ``L2_to_target`` field is left
NaN because no target profile is specified.

For new work on drug / tracer gradients use
``examples/tumor_chip_linear_gradient/`` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ooc_optimizer.config import load_config
from ooc_optimizer.optimization.bo_loop import BORunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(HERE / "config.yaml")
    # Use a single discrete configuration for the retained example to keep
    # the runtime bounded; switch to run_all_configurations() to sweep all
    # 8 v1 configurations.
    runner = BORunner(
        config=config,
        pillar_config="none",
        H=200.0,
        topology="opposing",  # v2 schema requires a topology; "opposing"
                              # with r_flow=0.5 reduces to the v1 single
                              # inlet in the limit delta_W → 0
    )
    result = runner.run()
    best = result["best_feasible"]
    if best is None:
        raise SystemExit("No feasible WSS-uniform geometry found")
    print(f"WSS-uniformity winner: L2={best['L2_to_target']}")
    print(f"params: {best['params']}")


if __name__ == "__main__":
    main()
