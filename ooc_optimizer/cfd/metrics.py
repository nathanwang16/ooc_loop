"""
Module 2.2 — Metric Extraction

Extracts optimization-relevant metrics from a completed OpenFOAM simulation.

Floor WSS is computed from the 2D depth-averaged velocity via the analytical
parabolic profile assumption:  τ_floor = 6 μ U_avg / H
(not from the OpenFOAM wallShearStress function object).
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def extract_metrics(
    case_dir: Path,
    H: float,
    mu: float,
) -> Dict[str, float]:
    """Extract WSS and flow metrics from a completed simulation.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory with converged results.
    H : float
        Channel height in m (for τ = 6μU/H).
    mu : float
        Dynamic viscosity in Pa·s.

    Returns
    -------
    metrics : dict
        cv_tau, tau_mean, tau_min, tau_max, f_dead, delta_p, converged.
    """
    raise NotImplementedError("Module 2.2 — metric extraction not yet implemented")


def _read_velocity_field(case_dir: Path) -> np.ndarray:
    """Parse the final-timestep U field from OpenFOAM output."""
    raise NotImplementedError


def _compute_floor_wss(U_avg: np.ndarray, H: float, mu: float) -> np.ndarray:
    """τ_floor = 6 μ U_avg / H for each cell on the culture floor."""
    raise NotImplementedError


def _compute_dead_fraction(U_field: np.ndarray, threshold_ratio: float = 0.1) -> float:
    """Fraction of floor area where velocity < threshold_ratio × mean velocity."""
    raise NotImplementedError


def _compute_pressure_drop(case_dir: Path) -> float:
    """Compute inlet-to-outlet pressure drop."""
    raise NotImplementedError
