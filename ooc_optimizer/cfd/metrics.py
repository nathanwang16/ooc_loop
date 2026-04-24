"""
Module 2.2 — Metric extraction (v2).

After both simpleFoam and scalarTransportFoam have run, read the converged
fields from the latest time directory and compute:

    L2_to_target      — primary v2 objective (relative L2 between simulated
                        and target concentration fields on the chamber floor)
    grad_sharpness    — diagnostic (mean |∇C| · L)
    monotonicity      — diagnostic (fraction of adjacent cells with consistent
                        sign of ∂C/∂axis; only meaningful for monotonic targets)
    tau_mean, tau_*, cv_tau, f_dead, delta_p, converged  — retained v1 WSS
                        metrics; used as constraints and sanity checks

Floor WSS is computed from the 2D depth-averaged velocity via the analytical
parabolic profile ``τ_floor = 6 μ U_avg / H``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ooc_optimizer.cfd.foam_parser import (
    find_latest_time,
    read_cell_centres,
    read_scalar_field,
    read_vector_field,
)
from ooc_optimizer.optimization.objectives import (
    TargetProfile,
    gradient_sharpness,
    l2_to_target,
    monotonicity_fraction,
)

logger = logging.getLogger(__name__)


def extract_v2_metrics(
    case_dir: Path,
    H: float,
    mu: float,
    *,
    chamber_length_m: float,
    chamber_width_m: float,
    target_profile: Optional[TargetProfile] = None,
) -> Dict[str, float]:
    """Extract the v2 metric set from a fully solved OpenFOAM case."""
    case_dir = Path(case_dir)

    try:
        latest_time = find_latest_time(case_dir)
        if latest_time is None:
            raise FileNotFoundError("No result time directory found.")

        # Momentum fields (first solve's latest time was overwritten by the
        # scalar solve, so read U from the latest dir which still contains U
        # because scalarTransportFoam writes it unchanged).
        u_file = latest_time / "U"
        U_field = read_vector_field(u_file)
        U_mag = np.linalg.norm(U_field, axis=1)
        centres = read_cell_centres(case_dir)

        # Restrict diagnostics to the "chamber interior" — the central 80% of
        # the chamber length, which excludes inlet/outlet taper regions.
        x = centres[:, 0]
        mask = (x > 0.1 * chamber_length_m) & (x < 0.9 * chamber_length_m)
        if not np.any(mask):
            mask = np.ones_like(U_mag, dtype=bool)

        # Momentum / WSS diagnostics (retained from v1).
        U_mag_dev = U_mag[mask]
        tau_floor = (6.0 * mu * U_mag_dev) / H
        tau_mean = float(np.mean(tau_floor))
        tau_std = float(np.std(tau_floor))
        cv_tau = tau_std / tau_mean if tau_mean > 0 else 999.0
        f_dead = _dead_fraction(U_mag_dev)
        delta_p = _pressure_drop(latest_time)

        # Scalar-field diagnostics.
        L2 = float("nan")
        grad_sharp = float("nan")
        mono = float("nan")
        C_field = None
        t_file = latest_time / "T"
        if t_file.exists():
            C_field = read_scalar_field(t_file)
            if len(C_field) != len(centres):
                raise ValueError("T / cell-centre length mismatch")
            C_dev = C_field[mask]
            centres_dev = centres[mask]

            if target_profile is not None:
                C_target = target_profile.evaluate(
                    centres_dev[:, 0],
                    centres_dev[:, 1],
                    L=chamber_length_m,
                    W=chamber_width_m,
                )
                L2 = l2_to_target(C_dev, C_target)
                mono_axis = "x"
                if target_profile.kind == "linear_gradient":
                    mono_axis = str(target_profile.params.get("axis", "x"))
                mono = monotonicity_fraction(C_dev, centres_dev, axis=mono_axis)

            grad_sharp = gradient_sharpness(C_dev, centres_dev, L=chamber_length_m)

        return {
            "L2_to_target": L2,
            "grad_sharpness": grad_sharp,
            "monotonicity": mono,
            "cv_tau": float(cv_tau),
            "tau_mean": tau_mean,
            "tau_min": float(np.min(tau_floor)) if tau_floor.size else 0.0,
            "tau_max": float(np.max(tau_floor)) if tau_floor.size else 0.0,
            "f_dead": f_dead,
            "delta_p": float(delta_p),
            "converged": True,
            "C_mean": float(np.mean(C_field)) if C_field is not None else float("nan"),
            "C_std": float(np.std(C_field)) if C_field is not None else float("nan"),
        }

    except Exception as exc:
        logger.error("v2 metric extraction failed: %s", exc, exc_info=True)
        return {
            "L2_to_target": 99.0,
            "grad_sharpness": 0.0,
            "monotonicity": 0.0,
            "cv_tau": 999.0,
            "tau_mean": 0.0,
            "tau_min": 0.0,
            "tau_max": 0.0,
            "f_dead": 1.0,
            "delta_p": 0.0,
            "converged": False,
            "C_mean": float("nan"),
            "C_std": float("nan"),
        }


def _dead_fraction(U_mag: np.ndarray, threshold_ratio: float = 0.1) -> float:
    if U_mag.size == 0:
        return 1.0
    u_mean = float(np.mean(U_mag))
    if u_mean == 0:
        return 1.0
    return float(np.sum(U_mag < threshold_ratio * u_mean) / U_mag.size)


def _pressure_drop(latest_time: Path) -> float:
    p_file = latest_time / "p"
    if not p_file.exists():
        return 0.0
    try:
        p = read_scalar_field(p_file)
        return float(np.max(p) - np.min(p))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Backward-compatible v1 entry point. The old BO orchestrator and legacy
# tests still call extract_metrics; route it through the v2 function with
# sensible defaults (no target ⇒ L2 stays NaN).
# ---------------------------------------------------------------------------


def extract_metrics(case_dir: Path, H: float, mu: float, L_mm: float = 20.0) -> Dict[str, float]:
    """Legacy v1 metric extraction (no scalar, no target).

    Kept for backward compatibility with the v1 tests and the retained
    WSS-uniformity example.  Do not use in the v2 BO loop — call
    ``extract_v2_metrics`` instead.
    """
    return extract_v2_metrics(
        case_dir=case_dir,
        H=H,
        mu=mu,
        chamber_length_m=L_mm * 1e-3,
        chamber_width_m=(L_mm / 10.0) * 1e-3,  # pragma: no cover (used only by legacy tests)
        target_profile=None,
    )
