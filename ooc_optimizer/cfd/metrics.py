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
from ooc_optimizer.cfd.foam_parser import (
    read_vector_field, 
    read_scalar_field, 
    find_latest_time
)

logger = logging.getLogger(__name__)


def extract_metrics(
    case_dir: Path,
    H: float,
    mu: float,
) -> Dict[str, float]:
    """
    Extract WSS and flow metrics from a completed simulation.

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
    case_dir = Path(case_dir)
    
    try:
        # 1. Read velocity field
        U_field = _read_velocity_field(case_dir)
        U_mag = np.linalg.norm(U_field, axis=1)

        # 2. Compute Shear Stress on the floor
        tau_floor = _compute_floor_wss(U_mag, H, mu)

        # 3. Compute Metrics
        tau_mean = np.mean(tau_floor)
        tau_std = np.std(tau_floor)
        cv_tau = tau_std / tau_mean if tau_mean > 0 else 999.0
        
        f_dead = _compute_dead_fraction(U_mag)
        delta_p = _compute_pressure_drop(case_dir)

        metrics = {
            "cv_tau": float(cv_tau),
            "tau_mean": float(tau_mean),
            "tau_min": float(np.min(tau_floor)),
            "tau_max": float(np.max(tau_floor)),
            "f_dead": float(f_dead),
            "delta_p": float(delta_p),
            "converged": True # Solver success is handled by the calling solver module
        }
        
        return metrics

    except Exception as e:
        logger.error(f"Failed to extract metrics from {case_dir}: {e}")
        return {
            "cv_tau": 999.0,
            "tau_mean": 0.0,
            "tau_min": 0.0,
            "tau_max": 0.0,
            "f_dead": 1.0,
            "delta_p": 0.0,
            "converged": False
        }


def _read_velocity_field(case_dir: Path) -> np.ndarray:
    """Parse the final-timestep U field from OpenFOAM output."""
    latest_time = find_latest_time(case_dir)
    if latest_time is None:
        raise FileNotFoundError(f"No result time directories found in {case_dir}")
    
    u_file = latest_time / "U"
    return read_vector_field(u_file)
    
    u_file = latest_time / "U"
    return read_vector_field(u_file)

def _read_velocity_field(case_dir: Path) -> np.ndarray:
    """Parse the final-timestep U field from OpenFOAM output."""
    latest_time = find_latest_time(case_dir)
    if latest_time is None:
        raise FileNotFoundError(f"No result time directories found in {case_dir}")
    u_file = latest_time / "U"
    return read_vector_field(u_file)


def _compute_floor_wss(U_avg: np.ndarray, H: float, mu: float) -> np.ndarray:
    """τ_floor = 6 μ U_avg / H for each cell on the culture floor."""
    # Based on the parabolic velocity profile for laminar flow between plates
    return (6.0 * mu * U_avg) / H


def _compute_dead_fraction(U_mag: np.ndarray, threshold_ratio: float = 0.1) -> float:
    """Fraction of floor area where velocity < threshold_ratio × mean velocity."""
    if U_mag.size == 0:
        return 1.0
    
    u_mean = np.mean(U_mag)
    if u_mean == 0:
        return 1.0
        
    dead_cells = np.sum(U_mag < (threshold_ratio * u_mean))
    return float(dead_cells / U_mag.size)


def _compute_pressure_drop(case_dir: Path) -> float:
    """Compute inlet-to-outlet pressure drop."""
    latest_time = find_latest_time(case_dir)
    if latest_time is None:
        return 0.0
        
    p_file = latest_time / "p"
    if not p_file.exists():
        return 0.0
        
    p_field = read_scalar_field(p_file)
    
    # In microfluidics with zero-gradient outlets and fixedValue inlets,
    # ΔP is roughly the max pressure at the inlet. 
    # For higher precision, one would parse the boundaryField for p, 
    # but internalField max is a robust proxy for optimization.
    return float(np.max(p_field) - np.min(p_field))
