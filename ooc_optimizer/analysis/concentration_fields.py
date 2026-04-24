"""
Module 3.2 — Concentration-field plotting (v2).

Replaces the v1 WSS-centric figure set.  Each plot function takes either a
converged OpenFOAM case directory or pre-extracted arrays, plus a
``TargetProfile`` for the BO-optimised configuration, and produces a
publication-quality matplotlib figure.

Figures produced here:
    * ``plot_concentration_contour`` — C(x, y) filled-contour map.
    * ``plot_residual_field``        — (C_achieved − C_target) difference map.
    * ``plot_centerline_profile``    — 1D C(x, W/2) overlay of achieved vs target.
    * ``plot_streamline_overlay``    — streamlines on top of the C(x, y) contour.

All plots use a fixed colour mapping (``viridis`` for fields, ``coolwarm``
for residuals) and write figures to the directory supplied by the caller;
nothing here hard-codes paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np

from ooc_optimizer.cfd.scalar import extract_concentration_field, frozen_flow_velocity
from ooc_optimizer.optimization.objectives import TargetProfile

matplotlib.use("Agg")  # non-interactive backend for batch scripts
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def _interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    L: float,
    W: float,
    nx: int = 200,
    ny: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered (x, y, values) onto a regular grid."""
    from scipy.interpolate import griddata

    xi = np.linspace(0.0, L, nx)
    yi = np.linspace(0.0, W, ny)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), values, (X, Y), method="linear")
    # Fall back to nearest for points outside the convex hull.
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        Z[nan_mask] = griddata((x, y), values, (X[nan_mask], Y[nan_mask]), method="nearest")
    return X, Y, Z


def plot_concentration_contour(
    case_dir: Path,
    *,
    L: float,
    W: float,
    output_path: Path,
    title: str = "Concentration field",
    levels: int = 20,
    cmap: str = "viridis",
) -> Path:
    """Plot the achieved concentration field on a regular (x, y) grid."""
    centres, C = extract_concentration_field(case_dir)
    X, Y, Z = _interpolate_to_grid(centres[:, 0], centres[:, 1], C, L=L, W=W)

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    cf = ax.contourf(X * 1e3, Y * 1e3, Z, levels=levels, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="C")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Concentration contour written to %s", output_path)
    return output_path


def plot_residual_field(
    case_dir: Path,
    target: TargetProfile,
    *,
    L: float,
    W: float,
    output_path: Path,
    title: str = "Residual (C_achieved − C_target)",
    cmap: str = "coolwarm",
) -> Path:
    """Plot the pointwise residual between achieved and target fields."""
    centres, C = extract_concentration_field(case_dir)
    C_target = target.evaluate(centres[:, 0], centres[:, 1], L=L, W=W)
    residual = C - C_target

    X, Y, Z = _interpolate_to_grid(centres[:, 0], centres[:, 1], residual, L=L, W=W)
    vmax = float(np.nanmax(np.abs(Z)))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    cf = ax.contourf(X * 1e3, Y * 1e3, Z, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="ΔC")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Residual field written to %s", output_path)
    return output_path


def plot_centerline_profile(
    case_dir: Path,
    target: TargetProfile,
    *,
    L: float,
    W: float,
    output_path: Path,
    title: str = "Centerline C(x, W/2)",
) -> Path:
    """Overlay 1D achieved and target centerline profiles."""
    centres, C = extract_concentration_field(case_dir)
    # Pick cells within a thin band around y = W/2.
    y_centre = W / 2.0
    band = 0.02 * W
    mask = np.abs(centres[:, 1] - y_centre) < band
    if mask.sum() < 5:
        # Nothing near the centerline; fall back to all cells at lowest |y - W/2|.
        order = np.argsort(np.abs(centres[:, 1] - y_centre))
        mask = np.zeros_like(centres[:, 1], dtype=bool)
        mask[order[:30]] = True

    x_c = centres[mask, 0]
    C_c = C[mask]
    order = np.argsort(x_c)
    x_c = x_c[order]
    C_c = C_c[order]

    x_ref = np.linspace(0.0, L, 200)
    C_target_ref = target.evaluate(x_ref, np.full_like(x_ref, y_centre), L=L, W=W)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(x_c * 1e3, C_c, "-", lw=1.5, label="Achieved", color="tab:blue")
    ax.plot(x_ref * 1e3, C_target_ref, "--", lw=1.5, label="Target", color="tab:red")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("C(x, W/2)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Centerline profile written to %s", output_path)
    return output_path


def plot_streamline_overlay(
    case_dir: Path,
    *,
    L: float,
    W: float,
    output_path: Path,
    title: str = "C contour + streamlines",
    cmap: str = "viridis",
    density: float = 1.2,
) -> Path:
    """Concentration contour with velocity streamlines superimposed."""
    centres, C = extract_concentration_field(case_dir)
    U = frozen_flow_velocity(case_dir)
    if U.shape[0] != centres.shape[0]:
        raise ValueError("U / centres length mismatch")

    X, Y, Z = _interpolate_to_grid(centres[:, 0], centres[:, 1], C, L=L, W=W)
    _, _, Ux = _interpolate_to_grid(centres[:, 0], centres[:, 1], U[:, 0], L=L, W=W)
    _, _, Uy = _interpolate_to_grid(centres[:, 0], centres[:, 1], U[:, 1], L=L, W=W)

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    cf = ax.contourf(X * 1e3, Y * 1e3, Z, levels=20, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.streamplot(X * 1e3, Y * 1e3, Ux, Uy, color="k", density=density, linewidth=0.4, arrowsize=0.6)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="C")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Streamline overlay written to %s", output_path)
    return output_path


def plot_bo_convergence(
    evaluations: list,
    *,
    output_path: Path,
    objective_key: str = "objective",
    title: str = "BO convergence",
) -> Path:
    """Cumulative-best-so-far objective vs iteration."""
    ys = [r[objective_key] for r in evaluations if objective_key in r]
    if not ys:
        raise ValueError(f"No evaluations contain '{objective_key}'")
    best_so_far = np.minimum.accumulate(ys)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(np.arange(1, len(ys) + 1), ys, "o", ms=3, color="tab:blue", label="Per-eval")
    ax.plot(np.arange(1, len(ys) + 1), best_so_far, "-", lw=1.5, color="tab:red", label="Best so far")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(objective_key)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_winner_grid(
    winners: list,
    L: float,
    W: float,
    output_dir: Path,
    *,
    target_profiles: Optional[list] = None,
) -> list:
    """Write a 3-row × 3-column grid (target / achieved / residual) for each winner.

    Each entry in ``winners`` is expected to be a dict with keys ``case_dir``
    and ``target_profile_spec`` (or an already-built ``TargetProfile``).
    """
    outputs = []
    for w in winners:
        case_dir = Path(w["case_dir"])
        target = w.get("target_profile")
        if not isinstance(target, TargetProfile):
            from ooc_optimizer.optimization.objectives import build_target_profile

            target = build_target_profile(dict(w["target_profile_spec"]))
        name = w.get("name", case_dir.name)
        d = Path(output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        outputs.append(
            {
                "contour": plot_concentration_contour(case_dir, L=L, W=W, output_path=d / "contour.png"),
                "residual": plot_residual_field(case_dir, target, L=L, W=W, output_path=d / "residual.png"),
                "centerline": plot_centerline_profile(case_dir, target, L=L, W=W, output_path=d / "centerline.png"),
            }
        )
    return outputs
