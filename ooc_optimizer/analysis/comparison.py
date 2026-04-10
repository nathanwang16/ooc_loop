"""
Cross-configuration comparison and constraint satisfaction plots.

Generates:
    - Parameter heatmaps across pillar configurations
    - Constraint satisfaction scatter plots (τ_mean vs. f_dead)
    - Summary table of best results per configuration
"""

import logging
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _is_feasible(metrics: Dict) -> bool:
    return (
        bool(metrics.get("converged", False))
        and bool(metrics.get("mesh_ok", True))
        and 0.5 <= float(metrics.get("tau_mean", 0.0)) <= 2.0
        and float(metrics.get("f_dead", 1.0)) <= 0.05
    )


def plot_constraint_scatter(
    evaluation_logs: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Scatter plot of τ_mean vs. f_dead, colored by feasibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tau_mean = []
    f_dead = []
    feasible = []
    for records in evaluation_logs.values():
        for rec in records:
            metrics = rec["metrics"]
            tau_mean.append(float(metrics.get("tau_mean", 0.0)))
            f_dead.append(float(metrics.get("f_dead", 1.0)))
            feasible.append(_is_feasible(metrics))

    tau_mean = np.array(tau_mean, dtype=float)
    f_dead = np.array(f_dead, dtype=float)
    feasible = np.array(feasible, dtype=bool)

    plt.figure(figsize=(8, 6))
    if tau_mean.size:
        plt.scatter(tau_mean[~feasible], f_dead[~feasible], s=12, alpha=0.6, label="Infeasible")
        plt.scatter(tau_mean[feasible], f_dead[feasible], s=16, alpha=0.8, label="Feasible")
    plt.axvline(0.5, color="k", linestyle="--", linewidth=1)
    plt.axvline(2.0, color="k", linestyle="--", linewidth=1)
    plt.axhline(0.05, color="k", linestyle="--", linewidth=1)
    plt.xlabel("τ_mean (Pa)")
    plt.ylabel("f_dead")
    plt.title("Constraint Satisfaction")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved constraint scatter to %s", output_path)
    return output_path


def plot_parameter_heatmap(
    best_per_config: Dict[str, Dict],
    output_path: Path,
) -> Path:
    """Heatmap of optimized parameters across configurations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not best_per_config:
        raise ValueError("best_per_config is empty; cannot plot heatmap")

    params = ["W", "d_p", "s_p", "theta", "Q"]
    configs = sorted(best_per_config.keys())
    data = np.full((len(configs), len(params)), np.nan, dtype=float)

    for i, cfg in enumerate(configs):
        cfg_params = best_per_config[cfg].get("params", {})
        for j, p in enumerate(params):
            if p in cfg_params:
                data[i, j] = float(cfg_params[p])

    plt.figure(figsize=(9, 5))
    img = plt.imshow(data, aspect="auto")
    plt.colorbar(img, label="Parameter value")
    plt.yticks(np.arange(len(configs)), configs)
    plt.xticks(np.arange(len(params)), params)
    plt.title("Best Parameters by Configuration")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved parameter heatmap to %s", output_path)
    return output_path


def generate_summary_table(
    best_per_config: Dict[str, Dict],
    output_path: Path,
) -> Path:
    """Write a summary CSV of best results per configuration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config",
        "cv_tau",
        "tau_mean",
        "f_dead",
        "converged",
        "W",
        "d_p",
        "s_p",
        "theta",
        "Q",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_name in sorted(best_per_config.keys()):
            entry = best_per_config[config_name]
            params = entry.get("params", {})
            metrics = entry.get("metrics", {})
            writer.writerow(
                {
                    "config": config_name,
                    "cv_tau": entry.get("cv_tau"),
                    "tau_mean": metrics.get("tau_mean"),
                    "f_dead": metrics.get("f_dead"),
                    "converged": metrics.get("converged"),
                    "W": params.get("W"),
                    "d_p": params.get("d_p"),
                    "s_p": params.get("s_p"),
                    "theta": params.get("theta"),
                    "Q": params.get("Q"),
                }
            )
    logger.info("Saved summary table to %s", output_path)
    return output_path
