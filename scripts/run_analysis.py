"""
Module 3.2 — Generate all analysis plots from optimization results.

Usage:
    python scripts/run_analysis.py --config configs/default_config.yaml \
        --log data/results/evaluations.jsonl \
        --output figures/
"""

import argparse
import logging
import sys
from pathlib import Path

from ooc_optimizer.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="Path to evaluation JSONL log"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures"), help="Output directory for figures"
    )
    args = parser.parse_args()

    raise NotImplementedError("Analysis script not yet implemented")


if __name__ == "__main__":
    main()
