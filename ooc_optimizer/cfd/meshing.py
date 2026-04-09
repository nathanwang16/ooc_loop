"""
Module 2.1 — Automated Meshing Pipeline

Takes a fluid domain STL and produces a valid OpenFOAM polyMesh directory.

Strategy:
    - No pillars  → blockMesh (structured quad, ~12k cells, 5–15 s)
    - With pillars → cfMesh / snappyHexMesh (~30k cells, 20–40 s)

On failure: logs error and returns None so the caller can assign a penalty.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_mesh(
    stl_path: Path,
    case_dir: Path,
    pillar_config: str,
    mesh_resolution: int,
) -> Optional[Path]:
    """
    Generate OpenFOAM mesh from fluid domain STL.

    Parameters
    ----------
    stl_path : Path
        Path to the fluid domain STL file.
    case_dir : Path
        OpenFOAM case directory (will be populated with polyMesh).
    pillar_config : str
        Pillar layout — determines meshing strategy.
    mesh_resolution : int
        Target cell count multiplier.

    Returns
    -------
    polyMesh path on success, None on failure.
    """
    case_dir = Path(case_dir)
    stl_path = Path(stl_path)
    
    # 1. Execute the appropriate mesher based on configuration
    if pillar_config.lower() == "none":
        logger.info("Executing blockMesh for baseline geometry.")
        success = _run_blockmesh(case_dir)
    else:
        logger.info(f"Executing cfMesh for {pillar_config} pillar geometry.")
        success = _run_cfmesh(case_dir, stl_path)

    if not success:
        logger.error("Meshing step failed.")
        return None

    # 2. Setup patch boundary types (ensures frontAndBack is set to 'empty' for 2D)
    try:
        _setup_patches(case_dir, pillar_config)
    except Exception as e:
        logger.error(f"Post-meshing patch setup failed: {e}")
        return None

    # 3. Validate mesh quality via checkMesh
    if not _validate_mesh(case_dir):
        logger.error("Mesh failed checkMesh validation.")
        return None

    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    
    # Return path on success, else None
    return poly_mesh_dir if poly_mesh_dir.exists() else None

def _run_blockmesh(case_dir: Path) -> bool:
    """Execute blockMesh for simple (no-pillar) geometries."""
    try:
        # Assumes constant/polyMesh/blockMeshDict is generated prior to this call
        subprocess.run(
            ["blockMesh"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ # Inherits OpenFOAM environment variables
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"blockMesh failed with error:\n{e.stderr}")
        return False

def _run_cfmesh(case_dir: Path, stl_path: Path) -> bool:
    """Execute cfMesh (cartesianMesh) for pillar geometries."""
    try:
        # Assumes system/meshDict points to stl_path
        subprocess.run(
            ["cartesianMesh"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ # Inherits OpenFOAM environment variables
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"cfMesh (cartesianMesh) failed with error:\n{e.stderr}")
        return False


def _validate_mesh(case_dir: Path) -> bool:
    """Run checkMesh and return True if no fatal errors."""
    try:
        result = subprocess.run(
            ["checkMesh", "-latestTime"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            env=os.environ
        )
        
        # Check for standard OpenFOAM success indicators
        if "Mesh OK" in result.stdout or "Failed 0 mesh checks" in result.stdout:
            logger.debug("checkMesh passed successfully.")
            return True
        else:
            logger.warning("checkMesh reported failures. Marking mesh as invalid.")
            return False
            
    except Exception as e:
        logger.error(f"Failed to execute checkMesh: {e}")
        return False


def _setup_patches(case_dir: Path, pillar_config: str) -> None:
    """Assign correct patch types (inlet, outlet, walls, floor, frontAndBack)."""
    # Uses changeDictionary to enforce patch types (e.g. 'empty' for 2D frontAndBack)
    # Requires system/changeDictionaryDict to be present in the case template.
    subprocess.run(
        ["changeDictionary"],
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=True,
        env=os.environ
    )
    logger.debug("Successfully applied changeDictionary to update patch types.")
