"""
Module 2.2 — CFD Run Automation

Orchestrates the full geometry → mesh → solve → extract pipeline into a single
callable function for the optimizer.

On failure (mesh error, solver divergence, timeout), returns penalty metrics
instead of crashing the loop.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from ooc_optimizer.cfd.meshing import MeshResult, generate_mesh
from ooc_optimizer.cfd.metrics import extract_metrics
from ooc_optimizer.cfd.verification import generate_blockmesh_dict
from ooc_optimizer.geometry import generate_pillar_obstacles_stl

logger = logging.getLogger(__name__)

# Standard penalty returned when a simulation step fails
PENALTY_METRICS = {
    "cv_tau": 999.0,
    "tau_mean": 0.0,
    "tau_min": 0.0,
    "tau_max": 0.0,
    "f_dead": 1.0,
    "delta_p": 0.0,
    "converged": False,
    "mesh_ok": False,
    "mesh_strategy_requested": None,
    "mesh_strategy_used": None,
    "mesh_checkmesh_ok": False,
    "mesh_used_fallback": False,
    "mesh_checkmesh_summary": {},
    "warnings": [],
    "case_dir": None,
}

SOLVER_TIMEOUT_S = 300  # 5 minutes

def _find_openfoam_prefix() -> Optional[str]:
    """Detect OpenFOAM invocation (Native vs macOS wrapper)."""
    if shutil.which("simpleFoam") is not None:
        return None
    for wrapper in ("openfoam2406", "openfoam2412", "openfoam2306"):
        if shutil.which(wrapper) is not None:
            return wrapper
    return None

def evaluate_cfd(
    params: Dict[str, float],
    pillar_config: str,
    H_um: float,
    config: dict,
) -> Dict[str, float]:
    """Run the full CFD evaluation pipeline for one parameter set."""
    H_m = H_um * 1e-6
    work_dir = Path(config["paths"]["work_dir"])
    template_dir = Path(config["paths"]["template_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup unique case directory
    case_name = f"run_{pillar_config}_{id(params)}"
    case_dir = work_dir / case_name
    
    try:
        _setup_case(template_dir, case_dir, params, H_m, pillar_config)
        _write_blockmesh_dict(case_dir=case_dir, params=params, pillar_config=pillar_config)
        
        # 2. Geometry and Meshing
        tri_surface_dir = case_dir / "constant" / "triSurface"
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        stl_path = tri_surface_dir / "pillars.stl"

        if pillar_config.lower() != "none":
            generate_pillar_obstacles_stl(params=params, pillar_config=pillar_config, output_path=stl_path)

        mesh_resolution = int(config.get("solver_settings", {}).get("mesh_resolution", 1))
        mesh_result = generate_mesh(
            stl_path,
            case_dir,
            pillar_config,
            mesh_resolution=mesh_resolution,
            mesh_options=config.get("solver_settings", {}),
        )
        if mesh_result is None:
            failure = dict(PENALTY_METRICS)
            failure["warnings"] = ["Mesh generation failed before solver execution."]
            failure["case_dir"] = str(case_dir)
            return failure

        # 3. Run Solver
        if not _run_simplefoam(case_dir):
            failure = dict(PENALTY_METRICS)
            failure["mesh_ok"] = mesh_result.mesh_ok
            failure["mesh_strategy_requested"] = mesh_result.strategy_requested
            failure["mesh_strategy_used"] = mesh_result.strategy_used
            failure["mesh_checkmesh_ok"] = mesh_result.checkmesh_ok
            failure["mesh_used_fallback"] = mesh_result.used_fallback
            failure["mesh_checkmesh_summary"] = dict(mesh_result.checkmesh_summary)
            failure["warnings"] = list(mesh_result.warnings) + ["simpleFoam failed to converge."]
            failure["case_dir"] = str(case_dir)
            return failure

        # 4. Generate Cell Center Data (NEW: Required for spatial metrics)
        prefix = _find_openfoam_prefix()
        post_cmd = ["postProcess", "-func", "writeCellCentres", "-latestTime"]
        if prefix:
            post_cmd = [prefix, "-c", " ".join(post_cmd)]
        
        subprocess.run(
            post_cmd, 
            cwd=case_dir, 
            check=True, 
            capture_output=True, 
            env=os.environ
        )

        # 5. Extract Metrics
        mu = config.get("physics", {}).get("mu", 1e-3)
        metrics = extract_metrics(case_dir, H_m, mu)
        metrics["mesh_ok"] = mesh_result.mesh_ok
        metrics["mesh_strategy_requested"] = mesh_result.strategy_requested
        metrics["mesh_strategy_used"] = mesh_result.strategy_used
        metrics["mesh_checkmesh_ok"] = mesh_result.checkmesh_ok
        metrics["mesh_used_fallback"] = mesh_result.used_fallback
        metrics["mesh_checkmesh_summary"] = dict(mesh_result.checkmesh_summary)
        metrics["warnings"] = list(mesh_result.warnings)
        metrics["case_dir"] = str(case_dir)
        if not mesh_result.mesh_ok:
            logger.warning(
                "Case %s has degraded mesh quality (strategy=%s, failed_checks=%s).",
                case_name,
                mesh_result.strategy_used,
                mesh_result.checkmesh_summary.get("failed_checks"),
            )
        return metrics

    except Exception as e:
        logger.error(f"CFD Pipeline failed for {case_name}: {e}")
        failure = dict(PENALTY_METRICS)
        failure["warnings"] = [f"CFD pipeline exception: {e}"]
        failure["case_dir"] = str(case_dir)
        return failure

def _setup_case(
    template_dir: Path,
    case_dir: Path,
    params: Dict[str, float],
    H: float,
    pillar_config: str,
) -> None:
    """Copy template and inject boundary conditions."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)
    
    # Calculate U = Q / (W * H)
    Q_m3s = (params["Q"] * 1e-9) / 60.0
    W_m = params["W"] * 1e-6
    u_inlet = Q_m3s / (W_m * H)

    u_file = case_dir / "0" / "U"
    if not u_file.exists():
        raise FileNotFoundError(f"Missing velocity file in template: {u_file}")
    content = u_file.read_text()
    if "INLET_VELOCITY" in content:
        u_file.write_text(content.replace("INLET_VELOCITY", f"{u_inlet:.6f}"))
    else:
        # Fallback path when template stores a concrete inlet value already.
        inlet_pattern = re.compile(
            r"(inlet\s*\{.*?value\s+uniform\s+\()([^)]+)(\);)",
            flags=re.DOTALL,
        )
        updated_content, count = inlet_pattern.subn(
            rf"\g<1>{u_inlet:.6f} 0 0\g<3>",
            content,
            count=1,
        )
        if count != 1:
            raise ValueError("Failed to locate inlet velocity value in template U file")
        u_file.write_text(updated_content)

    if pillar_config.lower() != "none":
        _set_front_and_back_symmetry(case_dir / "0" / "U")
        _set_front_and_back_symmetry(case_dir / "0" / "p")
        _ensure_patch_entry(
            file_path=u_file,
            patch_name="pillars",
            patch_block="""    pillars
    {
        type            noSlip;
    }
""",
        )
        p_file = case_dir / "0" / "p"
        _ensure_patch_entry(
            file_path=p_file,
            patch_name="pillars",
            patch_block="""    pillars
    {
        type            zeroGradient;
    }
""",
        )

def _run_simplefoam(case_dir: Path, timeout: int = SOLVER_TIMEOUT_S) -> bool:
    """Execute simpleFoam with prefix detection."""
    log_path = case_dir / "simpleFoam.log"
    prefix = _find_openfoam_prefix()
    cmd = ["simpleFoam"]
    if prefix:
        cmd = [prefix, "-c", "simpleFoam"]

    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
                env=os.environ
            )
        return _check_convergence(case_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _write_blockmesh_dict(case_dir: Path, params: Dict[str, float], pillar_config: str) -> None:
    """Generate system/blockMeshDict for no-pillar rectangular runs."""
    L_mm = 10.0
    W_mm = float(params["W"]) / 1000.0
    dz_mm = 0.01
    nx = 200
    ny = max(20, int(round(20.0 * W_mm)))
    bmd = generate_blockmesh_dict(L_mm=L_mm, W_mm=W_mm, dz_mm=dz_mm, nx=nx, ny=ny)
    # Match template boundary files that expect separate 'walls' and 'floor' patches.
    bmd = bmd.replace(
        "            walls\n"
        "            {\n"
        "                type wall;\n"
        "                faces (\n"
        "                    (1 5 4 0)\n"
        "                    (3 7 6 2)\n"
        "                );\n"
        "            }\n",
        "            walls\n"
        "            {\n"
        "                type wall;\n"
        "                faces (\n"
        "                    (1 5 4 0)\n"
        "                );\n"
        "            }\n"
        "            floor\n"
        "            {\n"
        "                type wall;\n"
        "                faces (\n"
        "                    (3 7 6 2)\n"
        "                );\n"
        "            }\n",
    )
    if pillar_config.lower() != "none":
        bmd = bmd.replace("type empty;", "type symmetry;")
    blockmesh_dict_path = case_dir / "system" / "blockMeshDict"
    blockmesh_dict_path.write_text(bmd)


def _ensure_patch_entry(file_path: Path, patch_name: str, patch_block: str) -> None:
    """Insert a boundary patch stanza if it's not already present."""
    if not file_path.exists():
        raise FileNotFoundError(f"Boundary file missing: {file_path}")
    content = file_path.read_text()
    if re.search(rf"\b{re.escape(patch_name)}\b\s*\{{", content):
        return
    insert_pos = content.rfind("}")
    if insert_pos == -1:
        raise ValueError(f"Malformed OpenFOAM boundary file: {file_path}")
    updated = content[:insert_pos] + patch_block + content[insert_pos:]
    file_path.write_text(updated)


def _set_front_and_back_symmetry(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Boundary file missing: {file_path}")
    content = file_path.read_text()
    updated = re.sub(
        r"(frontAndBack\s*\{[^}]*type\s+)(empty)(\s*;)",
        r"\1symmetry\3",
        content,
        flags=re.DOTALL,
    )
    file_path.write_text(updated)

def _check_convergence(case_dir: Path, threshold: float = 1e-4) -> bool:
    """Parse log file to ensure residuals are below threshold."""
    log_path = case_dir / "simpleFoam.log"
    if not log_path.exists():
        return False

    with open(log_path, "r") as f:
        lines = f.readlines()
    
    converged_u, converged_p = False, False
    residual_pattern = re.compile(r"final residual\s*=\s*([0-9eE\+\-\.]+)", re.IGNORECASE)
    for line in reversed(lines):
        if any(v in line for v in ["Solving for Ux", "Solving for Uy"]):
            match = residual_pattern.search(line)
            if match:
                res = float(match.group(1))
                if res < threshold:
                    converged_u = True
        
        if "Solving for p" in line:
            match = residual_pattern.search(line)
            if match:
                res = float(match.group(1))
                if res < threshold:
                    converged_p = True
        
        if converged_u and converged_p:
            return True
            
    return False
