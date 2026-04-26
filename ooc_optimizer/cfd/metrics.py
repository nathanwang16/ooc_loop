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
    rho_kg_m3: float,
    diffusivity_m2_s: float,
) -> Dict[str, float]:
    """Extract the v2 metric set from a fully solved OpenFOAM case.

    Diagnostic dimensionless numbers are appended for cross-topology comparison:

        Re              = rho * U_avg * D_h / mu               (laminar gate)
        Pe_streamwise   = U_avg * L_chamber / D                 (axial advection vs diffusion)
        Pe_crossstream  = U_avg * W_chamber / D                 (lateral mixing budget)
        aspect_ratio    = W_chamber / H                         (PDMS-collapse gate)
        R2_to_linear    = coefficient of determination from a 1D linear fit of the
                          cell-binned C-mean profile against the target axis;
                          NaN when target_profile is not a linear_gradient.

    rho_kg_m3 and diffusivity_m2_s are required (no defaults) per project rule.
    """
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

        # Dimensionless physics diagnostics.
        U_avg = float(np.mean(U_mag_dev)) if U_mag_dev.size else 0.0
        D_h = (2.0 * H * chamber_width_m) / (H + chamber_width_m) if (H + chamber_width_m) > 0 else 0.0
        Re = (rho_kg_m3 * U_avg * D_h) / mu if mu > 0 else float("nan")
        Pe_streamwise = (U_avg * chamber_length_m) / diffusivity_m2_s if diffusivity_m2_s > 0 else float("nan")
        Pe_crossstream = (U_avg * chamber_width_m) / diffusivity_m2_s if diffusivity_m2_s > 0 else float("nan")
        aspect_ratio = chamber_width_m / H if H > 0 else float("nan")

        # Scalar-field diagnostics.
        L2 = float("nan")
        grad_sharp = float("nan")
        mono = float("nan")
        R2_to_linear = float("nan")
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

                # R²-to-linear: only meaningful when target itself is a linear
                # gradient. Bin C along the target axis, fit y = a + b·xi, and
                # report the coefficient of determination.
                if target_profile.kind == "linear_gradient":
                    R2_to_linear = _r2_to_linear(
                        C_dev,
                        centres_dev,
                        axis=mono_axis,
                        L=chamber_length_m,
                        W=chamber_width_m,
                    )

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
            "Re": float(Re),
            "Pe_streamwise": float(Pe_streamwise),
            "Pe_crossstream": float(Pe_crossstream),
            "aspect_ratio": float(aspect_ratio),
            "R2_to_linear": float(R2_to_linear),
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
            "C_mean": 0.0,
            "C_std": 0.0,
            # Same finite-sentinel rule as solver.PENALTY_METRICS — see comment
            # there.  NaN here propagated into BoTorch and crashed the GP fit
            # on the opposing topology mid-run on 2026-04-26.
            "Re": 1.0e6,
            "Pe_streamwise": 0.0,
            "Pe_crossstream": 0.0,
            "aspect_ratio": 1.0e3,
            "R2_to_linear": 0.0,
        }


def _r2_to_linear(
    C_dev: np.ndarray,
    centres_dev: np.ndarray,
    *,
    axis: str,
    L: float,
    W: float,
    n_bins: int = 30,
) -> float:
    """Bin C along the target axis, fit a + b*xi via polyfit, return R²."""
    if axis == "y":
        coord = centres_dev[:, 1]
        extent = W
    else:
        coord = centres_dev[:, 0]
        extent = L
    if extent <= 0 or coord.size == 0:
        return float("nan")
    xi = coord / extent
    edges = np.linspace(xi.min(), xi.max(), n_bins + 1)
    bin_idx = np.digitize(xi, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centres_bin = 0.5 * (edges[:-1] + edges[1:])
    C_means: list = []
    xi_means: list = []
    for k in range(n_bins):
        sel = bin_idx == k
        if np.any(sel):
            C_means.append(float(np.mean(C_dev[sel])))
            xi_means.append(float(centres_bin[k]))
    if len(C_means) < 3:
        return float("nan")
    xi_arr = np.array(xi_means)
    C_arr = np.array(C_means)
    coeffs = np.polyfit(xi_arr, C_arr, 1)
    C_fit = np.polyval(coeffs, xi_arr)
    ss_res = float(np.sum((C_arr - C_fit) ** 2))
    ss_tot = float(np.sum((C_arr - C_arr.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


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

    Uses water-like rho/D (1000 kg/m³, 1e-10 m²/s) for the dimensionless
    diagnostics — these legacy callers do not exercise the BO path.
    """
    return extract_v2_metrics(
        case_dir=case_dir,
        H=H,
        mu=mu,
        chamber_length_m=L_mm * 1e-3,
        chamber_width_m=(L_mm / 10.0) * 1e-3,  # pragma: no cover (used only by legacy tests)
        target_profile=None,
        rho_kg_m3=1000.0,
        diffusivity_m2_s=1.0e-10,
    )
