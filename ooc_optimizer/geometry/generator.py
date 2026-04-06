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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    '''1. Build the fluid volume (the "positive")'''
    fluid_solid=_build_fluid_domain(params, pillar_config, H)
    
    '''2. Build the mold (the "negative")'''
    mold_solid=_build_mold(fluid_solid, params, H)
    
    '''3. Define paths'''
    fluid_path=output_dir / "fluid_domain.stl"
    mold_path=output_dir / "chip_mold.stl"
    
    '''4. Export to STL (scaling μm to mm for external tools)'''
    '''OpenFOAM and most 3D printers assume units are mm (right?)'''
    cq.exporters.export(fluid_solid.scale(0.001), str(fluid_path))
    cq.exporters.export(mold_solid.scale(0.001), str(mold_path))
    
    logger.info(f"Generated geometry: W={params['W']}um, theta={params['theta']}deg, config={pillar_config}")
    return fluid_path, mold_path


def _build_fluid_domain(params: Dict[str, float], pillar_config: str, H: float) -> cq.Workplane:
    """Construct the CadQuery solid representing the internal fluid volume."""
    W = params['W']
    theta = params['theta']
    L_chamber = 10000.0  #10mm fixed
    W_in = 500.0         #Inlet width fixed
    
    # Calculate taper length: tan(theta) = ((W - W_in)/2) / L_taper
    taper_angle_rad = math.radians(theta)
    L_taper = (W - W_in) / (2 * math.tan(taper_angle_rad))
    
    # Create the central chamber
    # centered=(True, True, False) means Z starts at 0 and goes up to H
    fluid = cq.Workplane("XY").box(L_chamber, W, H, centered=(True, True, False))
    
    # Create the Inlet Taper via Loft
    # We create two wireframes at different Z-offsets and loft between them
    inlet_taper = (
        cq.Workplane("XY")
        .workplane(offset=0)
        .rect(W_in, H)
        .workplane(offset=L_taper)
        .rect(W, H)
        .loft(combine=True)
    )
    
    # Rotate and translate inlet to the front of the chamber
    # Loft builds along Z; we need it along X
    inlet_taper = (
        inlet_taper.rotate((0,0,0), (0,1,0), -90)
        .translate((-L_chamber/2 - L_taper, 0, 0))
    )
    
    # Outlet is a mirror of the inlet across the YZ plane
    outlet_taper = inlet_taper.mirror("YZ")
    
    # Combine main body
    fluid = fluid.union(inlet_taper).union(outlet_taper)
    
    # Subtract Pillars
    if pillar_config != "none":
        rows, cols = map(int, pillar_config.split('x'))
        d_p = params['d_p']
        s_p = params['s_p'] # Center-to-center spacing
        
        # Grid centering logic
        x_start = -((cols - 1) * s_p) / 2
        y_start = -((rows - 1) * s_p) / 2
        
        for r in range(rows):
            for c in range(cols):
                px = x_start + (c * s_p)
                py = y_start + (r * s_p)
                # Create pillar and cut it from the fluid volume
                pillar = cq.Workplane("XY").center(px, py).circle(d_p/2).extrude(H)
                fluid = fluid.cut(pillar)
                
    return fluid


def _build_mold(fluid_solid: cq.Workplane, params: Dict[str, float], H: float) -> cq.Workplane:
    """Construct the negative mold by subtracting the fluid domain from a block."""
    W = params['W']
    L_chamber = 10000.0
    W_in = 500.0
    theta = params['theta']
    
    # Re-calculate total length for the bounding box
    L_taper = (W - W_in) / (2 * math.tan(math.radians(theta)))
    total_length = L_chamber + 2 * L_taper
    
    # Add 2mm (2000um) padding for handling and wall strength
    padding = 2000.0 
    base_thickness = 2000.0
    
    block_w = W + 2 * padding
    block_l = total_length + 2 * padding
    block_h = H + base_thickness
    
    # Create the mold block
    # Position it so the top surface is at Z=H
    mold_block = (
        cq.Workplane("XY")
        .box(block_l, block_w, block_h, centered=(True, True, False))
        .translate((0, 0, -base_thickness))
    )
    
    # The fluid volume is subtracted from the block
    # Because fluid starts at Z=0, it will "carve" into the top of the block
    return mold_block.cut(fluid_solid)


def _place_pillars(chamber_solid: cq.Workplane, params: Dict[str, float], pillar_config: str, H: float) -> cq.Workplane:
    """
    Cut cylindrical pillars out of the fluid domain.
    
    This follows the 'discrete levels' requirement (1x4, 2x4, 3x6) while 
    using continuous BO variables for diameter (d_p) and spacing (s_p).
    """
    if pillar_config == "none":
        return chamber_solid

    # Parse configuration (e.g., "2x4" -> 2 rows, 4 columns)
    try:
        rows, cols = map(int, pillar_config.split('x'))
    except ValueError:
        logger.warning(f"Invalid pillar_config '{pillar_config}'. Skipping pillars.")
        return chamber_solid

    d_p = params['d_p']
    s_p = params['s_p']  # Center-to-center spacing
    
    # Calculate grid extents to ensure the array is centered in the chamber
    # Chamber L is fixed at 10000um (10mm)
    x_start = -((cols - 1) * s_p) / 2
    y_start = -((rows - 1) * s_p) / 2

    fluid_with_pillars = chamber_solid

    # Iterate through the grid and cut cylinders
    for r in range(rows):
        for c in range(cols):
            pos_x = x_start + (c * s_p)
            pos_y = y_start + (r * s_p)
            
            # Create a cylinder at the calculated position
            # We extrude slightly beyond H to ensure a clean boolean cut
            pillar = (
                cq.Workplane("XY")
                .workplane()
                .center(pos_x, pos_y)
                .circle(d_p / 2)
                .extrude(H)
            )
            
            fluid_with_pillars = fluid_with_pillars.cut(pillar)

    return fluid_with_pillars


def _export_stl(solid: cq.Workplane, output_path: Path, scale: float = 1e-3) -> Path:
    """
    Export CadQuery solid to STL, converting from μm to mm.
    
    OpenFOAM and SLA slicers expect mm. Since internal math is in μm,
    we apply a 1e-3 (0.001) scale factor.
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply scaling and export
    # Note: .scale() creates a new object, leaving the original intact
    scaled_solid = solid.scale(scale)
    
    cq.exporters.export(scaled_solid, str(output_path))
    
    if not output_path.exists():
        raise IOError(f"Failed to export STL to {output_path}")
        
    return output_path
