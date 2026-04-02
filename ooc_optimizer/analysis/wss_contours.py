"""
WSS contour map generation.

Produces 2D color maps of τ_floor on the culture surface for baseline
vs. optimized geometries — the key paper figures.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_wss_contour(
    case_dir: Path,
    H: float,
    mu: float,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Generate a WSS contour map from a completed CFD case.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory with converged results.
    H : float
        Channel height in m.
    mu : float
        Dynamic viscosity in Pa·s.
    output_path : Path
        Path to save the figure.
    title : str, optional
        Figure title.

    Returns
    -------
    output_path : Path
    """
    raise NotImplementedError("Module 3.2 — WSS contour plotting not yet implemented")


def plot_side_by_side(
    baseline_case: Path,
    optimized_case: Path,
    H: float,
    mu: float,
    output_path: Path,
) -> Path:
    """Side-by-side WSS contour: baseline (high CV) vs. optimized (low CV)."""
    raise NotImplementedError
