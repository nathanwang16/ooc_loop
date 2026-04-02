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
    raise NotImplementedError("Module 3.2 — convergence plotting not yet implemented")


def plot_best_feasible_vs_iteration(
    evaluation_logs: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Plot best feasible CV(τ) found so far vs. iteration number."""
    raise NotImplementedError
