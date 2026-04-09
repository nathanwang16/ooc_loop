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
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
def _find_openfoam_prefix() -> Optional[str]:
    """
    Detect the OpenFOAM invocation method.
    Ensures compatibility with macOS .app wrappers and native Linux installs.
    """
    if shutil.which("blockMesh") is not None:
        return None

    # Check for common macOS homebrew/wrapper names from your verification script
    for wrapper in ("openfoam2406", "openfoam2412", "openfoam2306", "openfoam2312"):
        if shutil.which(wrapper) is not None:
            return wrapper

    return None

def generate_mesh(
    stl_path: Path,
    case_dir: Path,
    pillar_config: str,
    mesh_resolution: int,
) -> Optional[Path]:
    case_dir = Path(case_dir)
    stl_path = Path(stl_path)
    
    # 1. Execute the appropriate mesher based on configuration
    if pillar_config.lower() == "none":
        logger.info("Executing blockMesh for baseline geometry.")
        success = _run_blockmesh(case_dir)
    else:
        logger.info(f"Executing cfMesh (cartesianMesh) for {pillar_config} pillars.")
        success = _run_cfmesh(case_dir)

    if not success:
        logger.error("Meshing step failed.")
        return None

    # 2. Setup patch boundary types (Sets frontAndBack to 'empty' for 2D)
    try:
        _setup_patches(case_dir)
    except Exception as e:
        logger.error(f"Post-meshing patch setup failed: {e}")
        return None

    # 3. Validate mesh quality via checkMesh
    if not _validate_mesh(case_dir):
        logger.error("Mesh failed checkMesh validation.")
        return None

    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    return poly_mesh_dir if poly_mesh_dir.exists() else None
    # 3. Validate mesh quality via checkMesh
    if not _validate_mesh(case_dir):
        logger.error("Mesh failed checkMesh validation.")
        return None

    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    
    # Return path on success, else None
    return poly_mesh_dir if poly_mesh_dir.exists() else None

def _run_blockmesh(case_dir: Path) -> bool:
    """Execute blockMesh using the prefix logic."""
    prefix = _find_openfoam_prefix()
    cmd = ["blockMesh"]
    full_cmd = [prefix, "-c", " ".join(cmd)] if prefix else cmd
    
    try:
        subprocess.run(full_cmd, cwd=case_dir, check=True, capture_output=True, env=os.environ)
        return True
    except subprocess.CalledProcessError:
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

def _setup_patches(case_dir: Path) -> None:
    """
    Assign correct patch types via changeDictionary.
    Crucial for 2D depth-averaged simulations to ensure frontAndBack is 'empty'.
    """
    prefix = _find_openfoam_prefix()
    cmd = ["changeDictionary"]
    full_cmd = [prefix, "-c", " ".join(cmd)] if prefix else cmd
    
    subprocess.run(
        full_cmd,
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=True,
        env=os.environ
    )
    logger.debug("Successfully applied changeDictionary.")

def _validate_mesh(case_dir: Path) -> bool:
    """Run checkMesh and return True if no fatal errors."""
    prefix = _find_openfoam_prefix()
    cmd = ["checkMesh", "-latestTime"]
    full_cmd = [prefix, "-c", " ".join(cmd)] if prefix else cmd

    try:
        result = subprocess.run(full_cmd, cwd=case_dir, capture_output=True, text=True, env=os.environ)
        return "Mesh OK" in result.stdout or "Failed 0 mesh checks" in result.stdout
    except Exception:
        return False
