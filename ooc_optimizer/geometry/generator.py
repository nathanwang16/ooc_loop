"""
Parametric chip geometry generator using CadQuery.

Input:  parameter vector x = (W, d_p, s_p, theta, Q) plus discrete config
Output: fluid_domain.stl and chip_mold.stl

Units convention: all internal math in μm, convert to mm for STL export.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import cadquery as cq

from ooc_optimizer.geometry.validation import validate_geometry

logger = logging.getLogger(__name__)

VALID_PILLAR_CONFIGS = {"none", "1x4", "2x4", "3x6"}
L_CHAMBER_UM = 10000.0
W_INLET_UM = 500.0
MOLD_WALL_MM = 2.0
MOLD_BASE_MM = 2.0
PIN_DIAMETER_MM = 1.0
PIN_HEIGHT_MM = 1.0
PILLAR_DZ_MM = 0.01


def _um_to_mm(value_um: float) -> float:
    return value_um / 1000.0


def _um_to_m(value_um: float) -> float:
    return value_um * 1e-6


def _validate_params(params: Dict[str, float], pillar_config: str, H: float) -> None:
    required = {"W", "theta", "Q"}
    if pillar_config != "none":
        required.update({"d_p", "s_p"})
    missing = [k for k in sorted(required) if k not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    if pillar_config not in VALID_PILLAR_CONFIGS:
        raise ValueError(f"Invalid pillar_config '{pillar_config}'")
    if H <= 0:
        raise ValueError("H must be > 0")


def _pillar_grid_centers_um(W_um: float, pillar_config: str) -> Tuple[int, int, list[Tuple[float, float]]]:
    if pillar_config == "none":
        return 0, 0, []
    rows, cols = map(int, pillar_config.split("x"))
    centers = []
    for i in range(rows):
        for j in range(cols):
            cx = L_CHAMBER_UM * (j + 1) / (cols + 1) - (L_CHAMBER_UM / 2.0)
            cy = W_um * (i + 1) / (rows + 1) - (W_um / 2.0)
            centers.append((cx, cy))
    return rows, cols, centers


def _compute_taper_length_um(W_um: float, theta_deg: float) -> float:
    theta_rad = math.radians(theta_deg)
    if abs(math.tan(theta_rad)) < 1e-12:
        raise ValueError("theta leads to invalid taper length (tan(theta) ~ 0)")
    return abs((W_um - W_INLET_UM) / (2.0 * math.tan(theta_rad)))


def _build_fluid_domain(params: Dict[str, float], pillar_config: str, H_um: float) -> cq.Workplane:
    W_um = float(params["W"])
    theta_deg = float(params["theta"])
    H_mm = _um_to_mm(H_um)
    W_mm = _um_to_mm(W_um)
    W_in_mm = _um_to_mm(W_INLET_UM)
    L_chamber_mm = _um_to_mm(L_CHAMBER_UM)
    taper_mm = _um_to_mm(_compute_taper_length_um(W_um, theta_deg))

    x0 = -L_chamber_mm / 2.0
    x1 = L_chamber_mm / 2.0
    x_in = x0 - taper_mm
    x_out = x1 + taper_mm

    profile = [
        (x_in, -W_in_mm / 2.0),
        (x0, -W_mm / 2.0),
        (x1, -W_mm / 2.0),
        (x_out, -W_in_mm / 2.0),
        (x_out, W_in_mm / 2.0),
        (x1, W_mm / 2.0),
        (x0, W_mm / 2.0),
        (x_in, W_in_mm / 2.0),
    ]

    fluid = cq.Workplane("XY").polyline(profile).close().extrude(H_mm)

    if pillar_config != "none":
        d_p_mm = _um_to_mm(float(params["d_p"]))
        _, _, centers_um = _pillar_grid_centers_um(W_um, pillar_config)
        for cx_um, cy_um in centers_um:
            fluid = fluid.cut(
                cq.Workplane("XY")
                .center(_um_to_mm(cx_um), _um_to_mm(cy_um))
                .circle(d_p_mm / 2.0)
                .extrude(H_mm + 0.001)
            )
    return fluid


def _build_mold(fluid_solid: cq.Workplane) -> cq.Workplane:
    bbox = fluid_solid.val().BoundingBox()
    fluid_len = bbox.xlen
    fluid_wid = bbox.ylen
    fluid_h = bbox.zlen

    mold_len = fluid_len + (2.0 * MOLD_WALL_MM)
    mold_wid = fluid_wid + (2.0 * MOLD_WALL_MM)
    mold_h = fluid_h + MOLD_BASE_MM

    mold_block = (
        cq.Workplane("XY")
        .box(mold_len, mold_wid, mold_h, centered=(True, True, False))
        .translate((0, 0, -MOLD_BASE_MM))
    )
    mold = mold_block.cut(fluid_solid)

    # Add two alignment pins on opposite corners.
    pin_offset_x = (mold_len / 2.0) - MOLD_WALL_MM
    pin_offset_y = (mold_wid / 2.0) - MOLD_WALL_MM
    pin1 = (
        cq.Workplane("XY")
        .center(pin_offset_x, pin_offset_y)
        .circle(PIN_DIAMETER_MM / 2.0)
        .extrude(PIN_HEIGHT_MM)
    )
    pin2 = (
        cq.Workplane("XY")
        .center(-pin_offset_x, -pin_offset_y)
        .circle(PIN_DIAMETER_MM / 2.0)
        .extrude(PIN_HEIGHT_MM)
    )
    return mold.union(pin1).union(pin2)


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
    _validate_params(params, pillar_config, H)
    output_dir.mkdir(parents=True, exist_ok=True)

    fluid_solid = _build_fluid_domain(params, pillar_config, H)
    validation_errors = validate_geometry(params=params, pillar_config=pillar_config, H=H, solid=fluid_solid)
    if validation_errors:
        raise ValueError(f"Geometry validation failed: {validation_errors}")

    mold_solid = _build_mold(fluid_solid)

    fluid_path = output_dir / "fluid_domain.stl"
    mold_path = output_dir / "chip_mold.stl"

    cq.exporters.export(fluid_solid, str(fluid_path))
    cq.exporters.export(mold_solid, str(mold_path))

    logger.info(
        "Generated geometry for %s: W=%.1f um, theta=%.1f deg, H=%.1f um",
        pillar_config,
        float(params["W"]),
        float(params["theta"]),
        H,
    )
    return fluid_path, mold_path


def generate_pillar_obstacles_stl(
    params: Dict[str, float],
    pillar_config: str,
    output_path: Path,
    dz_mm: float = PILLAR_DZ_MM,
) -> Path:
    """Generate STL containing only pillar solids for snappyHexMesh carving.

    The pillar coordinate system matches blockMesh domain coordinates used by
    solver automation: x ∈ [0, L], y ∈ [0, W], z ∈ [0, dz].
    """
    if pillar_config == "none":
        raise ValueError("No pillar obstacles for pillar_config='none'")
    _validate_params(params, pillar_config, H=200.0)

    W_um = float(params["W"])
    d_p_m = _um_to_m(float(params["d_p"]))
    W_m = _um_to_m(W_um)
    L_m = _um_to_m(L_CHAMBER_UM)
    dz_m = dz_mm * 1e-3
    rows, cols, _ = _pillar_grid_centers_um(W_um, pillar_config)

    pillars = None
    for i in range(rows):
        for j in range(cols):
            cx_m = L_m * (j + 1) / (cols + 1)
            cy_m = W_m * (i + 1) / (rows + 1)
            cyl = (
                cq.Workplane("XY")
                .center(cx_m, cy_m)
                .circle(d_p_m / 2.0)
                .extrude(dz_m)
            )
            pillars = cyl if pillars is None else pillars.union(cyl)

    if pillars is None:
        raise ValueError(f"No pillars generated for config '{pillar_config}'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(pillars, str(output_path))
    return output_path
