"""
Module 1.2 — Parametric chip geometry generator (v2).

Inputs
------
params : dict
    Continuous parameters. The required keys depend on the pillar config and
    topology:
        ``W``       chamber width [μm]        (all)
        ``theta``   inlet taper angle [deg]   (all; used only when tapered)
        ``Q_total`` total volumetric flow rate [μL/min]  (all)
        ``r_flow``  Q_drug / Q_total in [0, 1]            (all)
        ``d_p``     pillar diameter [μm]      (pillar_config != "none")
        ``s_p``     pillar gap / spacing [μm] (pillar_config != "none")
        ``delta_W`` inlet separation normalized by W in [0, 0.5]  (topology="opposing")
pillar_config : str     one of {"none", "1x4", "2x4", "3x6"}
H : float               chamber height [μm], one of {200, 300}
topology : str          one of {"opposing", "same_side_Y", "asymmetric_lumen"}

Outputs
-------
fluid_stl_path : Path   watertight fluid-domain STL (for fabrication reference
                        and validation only — 2D CFD uses blockMesh).
mold_stl_path  : Path   mold STL with alignment pins.

Design notes
------------
The 2D CFD pipeline meshes directly from blockMeshDict (see
``ooc_optimizer.geometry.topology_blockmesh``) so the STL is not on the
simulation hot path.  For the ``asymmetric_lumen`` topology the STL is a best
representation of the intended fabricated chip, not a perfect CAD-from-
simulation round-trip.

Patch naming contract (v2)
--------------------------
All downstream tooling (meshing, solver, metrics) expects these patch names:
    inlet_drug, inlet_medium, outlet, walls, floor, frontAndBack.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cadquery as cq

from ooc_optimizer.geometry.validation import validate_geometry

logger = logging.getLogger(__name__)

VALID_PILLAR_CONFIGS = {"none", "1x4", "2x4", "3x6"}
VALID_TOPOLOGIES = {"opposing", "same_side_Y", "asymmetric_lumen"}

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


def _required_params(pillar_config: str, topology: str) -> Iterable[str]:
    required = {"W", "theta", "Q_total", "r_flow"}
    if pillar_config != "none":
        required.update({"d_p", "s_p"})
    if topology == "opposing":
        required.add("delta_W")
    return required


def _validate_params(params: Dict[str, float], pillar_config: str, H: float, topology: str) -> None:
    required = _required_params(pillar_config, topology)
    missing = [k for k in sorted(required) if k not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    if pillar_config not in VALID_PILLAR_CONFIGS:
        raise ValueError(f"Invalid pillar_config '{pillar_config}'")
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(f"Invalid topology '{topology}'")
    if H <= 0:
        raise ValueError("H must be > 0")
    r_flow = float(params["r_flow"])
    if not 0.0 <= r_flow <= 1.0:
        raise ValueError(f"r_flow must be in [0, 1], got {r_flow}")
    if topology == "opposing":
        dW = float(params["delta_W"])
        if not 0.05 <= dW <= 0.5:
            raise ValueError(f"delta_W must be in [0.05, 0.5], got {dW}")


def _pillar_grid_centers_um(W_um: float, pillar_config: str) -> Tuple[int, int, list[Tuple[float, float]]]:
    """Evenly spaced pillar grid, coordinates relative to chamber centre.

    Returned centres are in the same convention used by blockMesh: x ∈ [0, L]
    *after* adding L/2, i.e. the returned list is centred on the chamber and
    the CFD pipeline shifts it by (L/2, W/2) when writing the pillars STL.
    """
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


# ---------------------------------------------------------------------------
# Fluid-domain builders (CadQuery) — one per topology.
# These produce watertight solids used for fabrication moulds; 2D CFD is
# handled by ooc_optimizer.geometry.topology_blockmesh, which reads the same
# parameter set.
# ---------------------------------------------------------------------------


def _build_fluid_opposing(params: Dict[str, float], H_um: float) -> cq.Workplane:
    """Opposing-inlet topology: two symmetric short-side inlets at x = 0."""
    W_um = float(params["W"])
    theta_deg = float(params["theta"])
    delta_W = float(params["delta_W"])
    H_mm = _um_to_mm(H_um)
    W_mm = _um_to_mm(W_um)
    W_in_mm = _um_to_mm(W_INLET_UM)
    L_chamber_mm = _um_to_mm(L_CHAMBER_UM)
    taper_mm = _um_to_mm(_compute_taper_length_um(W_um, theta_deg))

    delta_mm = delta_W * W_mm

    x0 = -L_chamber_mm / 2.0
    x1 = L_chamber_mm / 2.0
    x_in = x0 - taper_mm
    x_out = x1 + taper_mm

    # Main chamber + outlet taper.
    chamber_profile = [
        (x0, -W_mm / 2.0),
        (x1, -W_mm / 2.0),
        (x_out, -W_in_mm / 2.0),
        (x_out, W_in_mm / 2.0),
        (x1, W_mm / 2.0),
        (x0, W_mm / 2.0),
    ]
    chamber = cq.Workplane("XY").polyline(chamber_profile).close().extrude(H_mm)

    # Drug inlet channel: centred at y = -delta_mm (below centreline), length
    # 2 mm, width W_INLET_UM, tapering into the chamber.
    inlet_ch_len_mm = 2.0
    drug_cy = -delta_mm
    medium_cy = delta_mm
    for cy in (drug_cy, medium_cy):
        inlet = (
            cq.Workplane("XY")
            .center(x0 - inlet_ch_len_mm / 2.0, cy)
            .rect(inlet_ch_len_mm, W_in_mm)
            .extrude(H_mm)
        )
        chamber = chamber.union(inlet)

    return chamber


def _build_fluid_same_side_Y(params: Dict[str, float], H_um: float) -> cq.Workplane:
    """Y-junction topology: two feed lines merge into a single chamber inlet."""
    W_um = float(params["W"])
    theta_deg = float(params["theta"])
    H_mm = _um_to_mm(H_um)
    W_mm = _um_to_mm(W_um)
    W_in_mm = _um_to_mm(W_INLET_UM)
    L_chamber_mm = _um_to_mm(L_CHAMBER_UM)
    taper_mm = _um_to_mm(_compute_taper_length_um(W_um, theta_deg))

    x0 = -L_chamber_mm / 2.0
    x1 = L_chamber_mm / 2.0
    x_out = x1 + taper_mm
    x_in = x0 - taper_mm

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
    chamber = cq.Workplane("XY").polyline(profile).close().extrude(H_mm)

    # Upstream Y-junction: two inlet arms meeting at x = x_in.
    y_arm_len_mm = 2.0
    half = W_in_mm / 2.0
    for y_sign in (-1.0, +1.0):
        arm = (
            cq.Workplane("XY")
            .center(x_in - y_arm_len_mm * 0.707, y_sign * (half + y_arm_len_mm * 0.707))
            .rect(y_arm_len_mm, W_in_mm)
            .rotate((0, 0, 0), (0, 0, 1), 45.0 * y_sign)
            .extrude(H_mm)
        )
        chamber = chamber.union(arm)
    return chamber


def _build_fluid_asymmetric_lumen(params: Dict[str, float], H_um: float) -> cq.Workplane:
    """Asymmetric-lumen topology: drug feeds via a side lumen, medium via x=0."""
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
    chamber = cq.Workplane("XY").polyline(profile).close().extrude(H_mm)

    # Side lumen along y = -W/2 wall, spanning 60% of chamber length, centred.
    lumen_len_mm = 0.6 * L_chamber_mm
    lumen_cy = -W_mm / 2.0 - W_in_mm / 2.0
    lumen = (
        cq.Workplane("XY")
        .center(0.0, lumen_cy)
        .rect(lumen_len_mm, W_in_mm)
        .extrude(H_mm)
    )
    chamber = chamber.union(lumen)
    return chamber


_TOPOLOGY_BUILDERS = {
    "opposing": _build_fluid_opposing,
    "same_side_Y": _build_fluid_same_side_Y,
    "asymmetric_lumen": _build_fluid_asymmetric_lumen,
}


def _carve_pillars(fluid: cq.Workplane, params: Dict[str, float], pillar_config: str, H_um: float) -> cq.Workplane:
    if pillar_config == "none":
        return fluid
    W_um = float(params["W"])
    d_p_mm = _um_to_mm(float(params["d_p"]))
    H_mm = _um_to_mm(H_um)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_chip(
    params: Dict[str, float],
    pillar_config: str,
    H: float,
    output_dir: Path,
    topology: str = "opposing",
) -> Tuple[Path, Path]:
    """Generate fluid domain and mold STLs for one (params, config, topology).

    Parameters
    ----------
    params : dict
        Continuous parameter vector. See module docstring for required keys.
    pillar_config : str
        Pillar grid configuration.
    H : float
        Chamber height [μm].
    output_dir : Path
        Directory into which STLs are written.
    topology : str
        Inlet topology, one of {"opposing", "same_side_Y", "asymmetric_lumen"}.

    Returns
    -------
    (fluid_stl_path, mold_stl_path)
    """
    _validate_params(params, pillar_config, H, topology)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fluid_solid = _TOPOLOGY_BUILDERS[topology](params, H)
    fluid_solid = _carve_pillars(fluid_solid, params, pillar_config, H)

    validation_errors = validate_geometry(
        params=params, pillar_config=pillar_config, H=H, solid=fluid_solid
    )
    if validation_errors:
        raise ValueError(f"Geometry validation failed: {validation_errors}")

    mold_solid = _build_mold(fluid_solid)

    fluid_path = output_dir / "fluid_domain.stl"
    mold_path = output_dir / "chip_mold.stl"
    cq.exporters.export(fluid_solid, str(fluid_path))
    cq.exporters.export(mold_solid, str(mold_path))

    logger.info(
        "Generated %s geometry: W=%.1f um, theta=%.1f deg, H=%.1f um, pillars=%s",
        topology, float(params["W"]), float(params["theta"]), H, pillar_config,
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
    # Use opposing topology's required params as the lowest common denominator
    # for the pillar STL (pillar placement does not depend on topology).
    _validate_params(
        {**params, "r_flow": 0.5, "delta_W": 0.2},
        pillar_config=pillar_config,
        H=200.0,
        topology="opposing",
    )

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
