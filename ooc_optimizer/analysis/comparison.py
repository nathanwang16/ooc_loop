"""
Cross-configuration comparison and constraint satisfaction plots.

Generates:
    - Parameter heatmaps across pillar configurations
    - Constraint satisfaction scatter plots (τ_mean vs. f_dead)
    - Summary table of best results per configuration
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_constraint_scatter(
    evaluation_logs: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Scatter plot of τ_mean vs. f_dead, colored by feasibility."""
    raise NotImplementedError("Module 3.2 — constraint scatter not yet implemented")


def plot_parameter_heatmap(
    best_per_config: Dict[str, Dict],
    output_path: Path,
) -> Path:
    """Heatmap of optimized parameters across configurations."""
    raise NotImplementedError


def generate_summary_table(
    best_per_config: Dict[str, Dict],
    output_path: Path,
) -> Path:
    """Write a summary CSV of best results per configuration."""
    raise NotImplementedError
