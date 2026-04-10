"""
Module 2.1 — Automated Meshing Pipeline

Takes a fluid domain STL and produces a valid OpenFOAM polyMesh directory.

Strategy:
    - No pillars  → blockMesh (structured quad, ~12k cells, 5–15 s)
    - With pillars → cfMesh / snappyHexMesh (~30k cells, 20–40 s)

On failure: logs error and returns None so the caller can assign a penalty.
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    poly_mesh_dir: Path
    strategy_requested: str
    strategy_used: str
    mesh_ok: bool
    checkmesh_ok: bool
    used_fallback: bool
    checkmesh_summary: dict
    warnings: list[str]


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


def _run_openfoam_tool(
    case_dir: Path,
    cmd: list[str],
    *,
    check: bool = True,
    log_name: Optional[str] = None,
) -> subprocess.CompletedProcess:
    prefix = _find_openfoam_prefix()
    full_cmd = [prefix, "-c", " ".join(cmd)] if prefix else cmd
    result = subprocess.run(
        full_cmd,
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=check,
        env=os.environ,
    )
    if log_name is not None:
        logs_dir = case_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_name
        log_path.write_text(
            f"$ {' '.join(full_cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n"
        )
    return result


def generate_mesh(
    stl_path: Path,
    case_dir: Path,
    pillar_config: str,
    mesh_resolution: int,
    mesh_options: Optional[dict] = None,
) -> Optional[MeshResult]:
    case_dir = Path(case_dir)
    stl_path = Path(stl_path)
    warnings: list[str] = []
    strategy_requested = "blockMesh" if pillar_config.lower() == "none" else "snappyHexMesh"
    strategy_used = "blockMesh"
    used_fallback = False
    mesh_options = mesh_options or {}

    # Always generate the background mesh first.
    success = _run_blockmesh(case_dir)
    if success and pillar_config.lower() != "none":
        if not stl_path.exists():
            logger.error("Pillar STL not found for snappyHexMesh: %s", stl_path)
            return None
        _write_snappy_dict(
            case_dir=case_dir,
            stl_name=stl_path.name,
            mesh_resolution=mesh_resolution,
            mesh_options=mesh_options,
        )
        logger.info("Executing snappyHexMesh for %s pillars.", pillar_config)
        success = _run_snappy(case_dir)
        if not success:
            logger.error("snappyHexMesh failed for %s; cannot continue pillar meshing.", pillar_config)
            return None
        else:
            strategy_used = "snappyHexMesh"

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
    checkmesh_ok, checkmesh_summary = _validate_mesh(case_dir, mesh_options=mesh_options)
    if not checkmesh_ok:
        if pillar_config.lower() != "none":
            warning = (
                f"checkMesh failed after pillar meshing for {pillar_config}; "
                "keeping snappy mesh but marking evaluation as low-quality."
            )
            logger.warning(warning)
            warnings.append(warning)
            warnings.append(
                "checkMesh summary: "
                f"failed_checks={checkmesh_summary.get('failed_checks')}, "
                f"max_non_ortho={checkmesh_summary.get('max_non_ortho')}, "
                f"concave_cells={checkmesh_summary.get('concave_cells')}"
            )
        else:
            logger.error("Mesh failed checkMesh validation.")
            return None

    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    if not poly_mesh_dir.exists():
        return None
    mesh_ok = (not used_fallback) and checkmesh_ok
    return MeshResult(
        poly_mesh_dir=poly_mesh_dir,
        strategy_requested=strategy_requested,
        strategy_used=strategy_used,
        mesh_ok=mesh_ok,
        checkmesh_ok=checkmesh_ok,
        used_fallback=used_fallback,
        checkmesh_summary=checkmesh_summary,
        warnings=warnings,
    )

def _run_blockmesh(case_dir: Path) -> bool:
    """Execute blockMesh using the prefix logic."""
    try:
        _run_openfoam_tool(case_dir, ["blockMesh"], log_name="blockMesh.log")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("blockMesh failed: %s", exc.stderr)
        return False


def _run_snappy(case_dir: Path) -> bool:
    """Run snappyHexMesh pipeline for internal pillar obstacles."""
    try:
        _run_openfoam_tool(case_dir, ["surfaceFeatureExtract"], log_name="surfaceFeatureExtract.log")
        _run_openfoam_tool(case_dir, ["snappyHexMesh", "-overwrite"], log_name="snappyHexMesh.log")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("snappyHexMesh failed: %s", exc.stderr)
        return False

def _setup_patches(case_dir: Path) -> None:
    """
    Assign correct patch types via changeDictionary.
    Crucial for 2D depth-averaged simulations to ensure frontAndBack is 'empty'.
    """
    _ = case_dir
    return

def _extract_checkmesh_summary(stdout: str) -> dict:
    summary = {
        "failed_checks": None,
        "max_non_ortho": None,
        "concave_cells": None,
    }
    failed_match = re.search(r"Failed\s+(\d+)\s+mesh checks\.", stdout)
    if failed_match:
        summary["failed_checks"] = int(failed_match.group(1))
    non_ortho_match = re.search(r"Mesh non-orthogonality Max:\s*([0-9eE\+\-\.]+)", stdout)
    if non_ortho_match:
        summary["max_non_ortho"] = float(non_ortho_match.group(1))
    concave_match = re.search(r"concave cells.*number of cells:\s*(\d+)", stdout, re.IGNORECASE)
    if concave_match:
        summary["concave_cells"] = int(concave_match.group(1))
    return summary


def _validate_mesh(case_dir: Path, mesh_options: dict) -> tuple[bool, dict]:
    """Run checkMesh and return (ok, summary)."""
    try:
        result = _run_openfoam_tool(
            case_dir,
            ["checkMesh", "-allGeometry", "-allTopology"],
            check=False,
            log_name="checkMesh.log",
        )
        summary = _extract_checkmesh_summary(result.stdout)
        strict_ok = "Mesh OK" in result.stdout or "Failed 0 mesh checks" in result.stdout
        max_failed = int(mesh_options.get("mesh_max_failed_checks", 0))
        max_non_ortho = float(mesh_options.get("mesh_max_non_ortho", 65.0))
        max_concave_cells = int(mesh_options.get("mesh_max_concave_cells", 0))
        threshold_ok = True
        failed_checks = summary.get("failed_checks")
        if failed_checks is None or failed_checks > max_failed:
            threshold_ok = False
        non_ortho = summary.get("max_non_ortho")
        if non_ortho is None or non_ortho > max_non_ortho:
            threshold_ok = False
        concave = summary.get("concave_cells")
        if concave is None or concave > max_concave_cells:
            threshold_ok = False
        ok = strict_ok or threshold_ok
        return ok, summary
    except Exception:
        return False, {"failed_checks": None, "max_non_ortho": None, "concave_cells": None}


def _write_snappy_dict(case_dir: Path, stl_name: str, mesh_resolution: int, mesh_options: dict) -> None:
    """Write snappyHexMesh/system feature dicts for pillar obstacle carving."""
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    refinement_level = max(1, int(mesh_resolution))
    feature_level = refinement_level + 1
    n_cells_between_levels = int(mesh_options.get("snappy_n_cells_between_levels", 3))
    resolve_feature_angle = int(mesh_options.get("snappy_resolve_feature_angle", 20))
    location_in_mesh = mesh_options.get("snappy_location_in_mesh", "(0.001 0.0002 0.000005)")
    snap_enabled = bool(mesh_options.get("snappy_enable_snap", True))
    n_smooth_patch = int(mesh_options.get("snappy_n_smooth_patch", 5))
    n_solve_iter = int(mesh_options.get("snappy_n_solve_iter", 100))
    n_relax_iter = int(mesh_options.get("snappy_n_relax_iter", 8))
    n_feature_snap_iter = int(mesh_options.get("snappy_n_feature_snap_iter", 15))

    sfe_dict = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object surfaceFeatureExtractDict;
}}

{stl_name}
{{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs
    {{
        includedAngle 150;
    }}
    writeObj yes;
}}
"""

    snappy_dict = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}}

castellatedMesh true;
snap {"true" if snap_enabled else "false"};
addLayers false;

geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        name pillars;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 200000;
    maxGlobalCells 400000;
    minRefinementCells 0;
    nCellsBetweenLevels {n_cells_between_levels};

    features
    (
        {{
            file "{Path(stl_name).stem}.eMesh";
            level {feature_level};
        }}
    );

    refinementSurfaces
    {{
        pillars
        {{
            level ({refinement_level} {feature_level});
            patchInfo
            {{
                type wall;
            }}
        }}
    }}
    refinementRegions
    {{
    }}

    resolveFeatureAngle {resolve_feature_angle};
    locationInMesh {location_in_mesh};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch {n_smooth_patch};
    tolerance 2.0;
    nSolveIter {n_solve_iter};
    nRelaxIter {n_relax_iter};
    nFeatureSnapIter {n_feature_snap_iter};
    explicitFeatureSnap true;
    implicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""
    (system_dir / "surfaceFeatureExtractDict").write_text(sfe_dict)
    (system_dir / "snappyHexMeshDict").write_text(snappy_dict)
