"""
WSS contour map generation.

Produces 2D color maps of τ_floor on the culture surface for baseline
vs. optimized geometries — the key paper figures.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from ooc_optimizer.cfd.foam_parser import find_latest_time, read_cell_centres, read_vector_field

logger = logging.getLogger(__name__)


def _load_wss_field(case_dir: Path, H: float, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latest = find_latest_time(case_dir)
    if latest is None:
        raise FileNotFoundError(f"No converged time directory found in {case_dir}")
    U = read_vector_field(latest / "U")
    C = read_cell_centres(case_dir)
    if U.shape[0] != C.shape[0]:
        raise ValueError("U and C arrays have different cell counts")
    U_mag = np.linalg.norm(U[:, :2], axis=1)
    tau = 6.0 * mu * U_mag / H
    return C[:, 0], C[:, 1], tau


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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x, y, tau = _load_wss_field(case_dir=Path(case_dir), H=H, mu=mu)
    triang = mtri.Triangulation(x, y)

    plt.figure(figsize=(7, 3.5))
    contour = plt.tricontourf(triang, tau, levels=40)
    plt.colorbar(contour, label="τ_floor (Pa)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title or f"WSS Contour: {Path(case_dir).name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logger.info("Saved WSS contour to %s", output_path)
    return output_path


def plot_side_by_side(
    baseline_case: Path,
    optimized_case: Path,
    H: float,
    mu: float,
    output_path: Path,
) -> Path:
    """Side-by-side WSS contour: baseline (high CV) vs. optimized (low CV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xb, yb, taub = _load_wss_field(case_dir=Path(baseline_case), H=H, mu=mu)
    xo, yo, tauo = _load_wss_field(case_dir=Path(optimized_case), H=H, mu=mu)

    vmin = float(min(np.min(taub), np.min(tauo)))
    vmax = float(max(np.max(taub), np.max(tauo)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    t1 = mtri.Triangulation(xb, yb)
    t2 = mtri.Triangulation(xo, yo)
    c1 = axes[0].tricontourf(t1, taub, levels=40, vmin=vmin, vmax=vmax)
    axes[0].set_title("Baseline")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].tricontourf(t2, tauo, levels=40, vmin=vmin, vmax=vmax)
    axes[1].set_title("Optimized")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(c1, ax=axes, label="τ_floor (Pa)")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    logger.info("Saved baseline-vs-optimized contour to %s", output_path)
    return output_path
