"""
Parametric chip geometry generator using CadQuery.

Input:  parameter vector x = (W, d_p, s_p, theta, Q) plus discrete config
Output: fluid_domain.stl and chip_mold.stl

Units convention: all internal math in μm, convert to mm for STL export.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def generate_chip(
    params: Dict[str, float],
    pillar_config: str,
    H: float,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate fluid domain and mold STL files for a given parameter set.

    Parameters
    ----------
    params : dict
        Continuous parameters: W, d_p, s_p, theta, Q.
    pillar_config : str
        One of {"none", "1x4", "2x4", "3x6"}.
    H : float
        Chamber height in μm (200 or 300).
    output_dir : Path
        Directory to write STL files into.

    Returns
    -------
    (fluid_stl_path, mold_stl_path) : tuple of Path

    Raises
    ------
    ValueError
        If any required parameter is missing or out of bounds.
    """
    raise NotImplementedError("Module 1.2 — geometry generation not yet implemented")


def _build_fluid_domain(params: Dict[str, float], pillar_config: str, H: float):
    """Construct the CadQuery solid representing the internal fluid volume."""
    raise NotImplementedError


def _build_mold(fluid_solid, H: float, wall_thickness: float = 2000.0):
    """Subtract fluid domain from bounding box to produce the casting mold.

    wall_thickness is in μm (default 2 mm = 2000 μm).
    """
    raise NotImplementedError


def _place_pillars(chamber_solid, params: Dict[str, float], pillar_config: str, H: float):
    """Cut cylindrical pillars out of the fluid domain."""
    raise NotImplementedError


def _export_stl(solid, output_path: Path, scale: float = 1e-3) -> Path:
    """Export CadQuery solid to STL, converting from μm to mm."""
    raise NotImplementedError
