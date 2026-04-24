"""
Module 3.3 — Interpretability analysis driver.

Walks through the BO state directories produced by Module 3.1 (one per
(topology, pillar_config, H) combination that completed), runs Sobol + GP
gradient + tolerance-interval analyses on each, and writes a per-run
``summary.json`` plus the aggregated design-heuristics markdown.

Usage:
    python scripts/run_interpretability.py --results-dir data/results
    python scripts/run_interpretability.py --state-dir data/results/bo_opposing_none_H200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from ooc_optimizer.interpretability import analyse_winner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interpretability")


def _iter_state_dirs(results_dir: Path) -> List[Path]:
    return [p for p in sorted(results_dir.glob("bo_*")) if (p / "evaluations.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--state-dir", type=Path, help="Single BO state directory to analyse")
    group.add_argument(
        "--results-dir",
        type=Path,
        help="Parent directory — all bo_*/evaluations.json under it are analysed",
    )
    parser.add_argument("--sobol-n", type=int, default=1024)
    parser.add_argument("--loss-tol", type=float, default=0.1)
    args = parser.parse_args()

    if args.state_dir:
        targets = [args.state_dir]
    else:
        targets = _iter_state_dirs(args.results_dir)
        if not targets:
            raise SystemExit(f"No bo_* state directories under {args.results_dir}")

    all_summaries = []
    for sd in targets:
        logger.info("Analysing %s", sd)
        try:
            summary = analyse_winner(
                sd,
                sobol_n_samples=args.sobol_n,
                tolerance_loss_tolerance=args.loss_tol,
            )
            all_summaries.append({"state_dir": str(sd), "summary": summary})
        except Exception as exc:
            logger.error("Failed for %s: %s", sd, exc, exc_info=True)

    out = (targets[0].parent if args.results_dir else args.state_dir.parent) / "interpretability_index.json"
    with open(out, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info("Aggregate interpretability index written to %s", out)


if __name__ == "__main__":
    main()
