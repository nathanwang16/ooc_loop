"""
BO convergence curve plotting.

Generates CV(τ) vs. iteration plots for each of the 8 configurations,
showing the improvement from Sobol initialization through BO.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _extract_cv_series(records: List[Dict]) -> np.ndarray:
    return np.array([float(r["metrics"]["cv_tau"]) for r in records], dtype=float)


def _is_feasible(record: Dict) -> bool:
    metrics = record["metrics"]
    return (
        bool(metrics.get("converged", False))
        and bool(metrics.get("mesh_ok", True))
        and 0.5 <= float(metrics.get("tau_mean", 0.0)) <= 2.0
        and float(metrics.get("f_dead", 1.0)) <= 0.05
    )


def _best_feasible_curve(records: List[Dict]) -> np.ndarray:
    best = np.inf
    curve = []
    for rec in records:
        if _is_feasible(rec):
            best = min(best, float(rec["metrics"]["cv_tau"]))
        curve.append(best if np.isfinite(best) else np.nan)
    return np.array(curve, dtype=float)


def plot_convergence_curves(
    evaluation_logs: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Plot convergence curves for all configurations on one figure.

    Parameters
    ----------
    evaluation_logs : dict
        Keys are config names (e.g. "none_H200"), values are lists of
        evaluation records in chronological order.
    output_path : Path
        Path to save the figure.

    Returns
    -------
    output_path : Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for config_name, records in sorted(evaluation_logs.items()):
        if not records:
            continue
        cv = _extract_cv_series(records)
        best_so_far = np.minimum.accumulate(cv)
        x = np.arange(1, len(cv) + 1)
        plt.plot(x, best_so_far, label=config_name)

    plt.xlabel("Evaluation")
    plt.ylabel("Best CV(τ) so far")
    plt.title("BO Convergence Curves")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved convergence plot to %s", output_path)
    return output_path


def plot_best_feasible_vs_iteration(
    evaluation_logs: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Plot best feasible CV(τ) found so far vs. iteration number."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for config_name, records in sorted(evaluation_logs.items()):
        if not records:
            continue
        curve = _best_feasible_curve(records)
        x = np.arange(1, len(curve) + 1)
        plt.plot(x, curve, label=config_name)

    plt.xlabel("Evaluation")
    plt.ylabel("Best feasible CV(τ) so far")
    plt.title("Best Feasible Objective vs. Iteration")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved feasible convergence plot to %s", output_path)
    return output_path
