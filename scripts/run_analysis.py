"""
Module 3.2 — Generate all analysis plots from optimization results.

Usage:
    python scripts/run_analysis.py --config configs/default_config.yaml \
        --log data/results/evaluations.jsonl \
        --output figures/
"""

import argparse
import json
import logging
from pathlib import Path

from ooc_optimizer.config import load_config
from ooc_optimizer.analysis.comparison import (
    generate_summary_table,
    plot_constraint_scatter,
    plot_parameter_heatmap,
)
from ooc_optimizer.analysis.convergence import (
    plot_best_feasible_vs_iteration,
    plot_convergence_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_logs(log_path: Path):
    grouped = {}
    if log_path.exists():
        records = _load_jsonl(log_path)
        if records:
            grouped[log_path.stem] = records
        return grouped

    # Accept prefix input to collect per-config logs produced by BORunner.
    candidate_dir = log_path.parent if log_path.parent.exists() else Path(".")
    prefix = log_path.stem
    for p in sorted(candidate_dir.glob(f"{prefix}_*_H*.jsonl")):
        grouped[p.stem] = _load_jsonl(p)
    return grouped


def _best_per_config(evaluation_logs):
    best = {}
    for cfg, records in evaluation_logs.items():
        feasible = [
            r for r in records
            if r["metrics"].get("converged", False)
            and r["metrics"].get("mesh_ok", True)
            and 0.5 <= float(r["metrics"].get("tau_mean", 0.0)) <= 2.0
            and float(r["metrics"].get("f_dead", 1.0)) <= 0.05
        ]
        if not feasible:
            continue
        winner = min(feasible, key=lambda r: float(r["metrics"]["cv_tau"]))
        best[cfg] = {
            "cv_tau": float(winner["metrics"]["cv_tau"]),
            "params": winner.get("params", {}),
            "metrics": winner.get("metrics", {}),
        }
    return best


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

    _ = load_config(args.config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_logs = _group_logs(args.log)
    if not evaluation_logs:
        raise FileNotFoundError(
            f"No evaluation logs found for '{args.log}'. "
            "Provide an existing JSONL file or a prefix for per-config logs."
        )

    plot_convergence_curves(
        evaluation_logs=evaluation_logs,
        output_path=output_dir / "convergence_curves.png",
    )
    plot_best_feasible_vs_iteration(
        evaluation_logs=evaluation_logs,
        output_path=output_dir / "best_feasible_convergence.png",
    )
    plot_constraint_scatter(
        evaluation_logs=evaluation_logs,
        output_path=output_dir / "constraint_scatter.png",
    )

    best_cfg = _best_per_config(evaluation_logs)
    if best_cfg:
        plot_parameter_heatmap(
            best_per_config=best_cfg,
            output_path=output_dir / "parameter_heatmap.png",
        )
        generate_summary_table(
            best_per_config=best_cfg,
            output_path=output_dir / "best_summary.csv",
        )
    else:
        logger.warning("No feasible records found; skipped heatmap and summary table.")


if __name__ == "__main__":
    main()
