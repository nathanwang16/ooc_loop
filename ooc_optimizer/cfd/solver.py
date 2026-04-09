"""
Module 2.2 — CFD Run Automation

Orchestrates the full geometry → mesh → solve → extract pipeline into a single
callable function for the optimizer.

On failure (mesh error, solver divergence, timeout), returns penalty metrics
instead of crashing the loop.
"""

"""
Module 2.2 — CFD Run Automation

Orchestrates the full geometry → mesh → solve → extract pipeline.
Incorporates OpenFOAM environment detection and coordinate extraction.
"""

import logging
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from ooc_optimizer.cfd.meshing import generate_mesh
from ooc_optimizer.cfd.metrics import extract_metrics
from ooc_optimizer.geometry import generate_chip

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
    
    # 1. Setup unique case directory
    case_name = f"run_{pillar_config}_{id(params)}"
    case_dir = work_dir / case_name
    
    try:
        _setup_case(template_dir, case_dir, params, H_m)
        
        # 2. Geometry and Meshing
        stl_path = case_dir / "constant" / "triSurface" / "domain.stl"
        generate_chip(params, pillar_config, H_um, stl_path)
        
        mesh_path = generate_mesh(stl_path, case_dir, pillar_config, mesh_resolution=1)
        if mesh_path is None:
            return PENALTY_METRICS

        # 3. Run Solver
        if not _run_simplefoam(case_dir):
            return PENALTY_METRICS

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
        return extract_metrics(case_dir, H_m, mu)

    except Exception as e:
        logger.error(f"CFD Pipeline failed for {case_name}: {e}")
        return PENALTY_METRICS

def _setup_case(template_dir: Path, case_dir: Path, params: Dict[str, float], H: float) -> None:
    """Copy template and inject boundary conditions."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)
    
    # Calculate U = Q / (W * H)
    Q_m3s = (params["Q"] * 1e-9) / 60.0
    W_m = params["W"] * 1e-6
    u_inlet = Q_m3s / (W_m * H)

    u_file = case_dir / "0" / "U"
    subprocess.run(
        ["sed", "-i", f"s/INLET_VELOCITY/{u_inlet:.6f}/g", str(u_file)],
        check=True
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

def _check_convergence(case_dir: Path, threshold: float = 1e-4) -> bool:
    """Parse log file to ensure residuals are below threshold."""
    log_path = case_dir / "simpleFoam.log"
    if not log_path.exists():
        return False

    with open(log_path, "r") as f:
        lines = f.readlines()
    
    converged_u, converged_p = False, False
    for line in reversed(lines):
        if any(v in line for v in ["Solving for Ux", "Solving for Uy"]):
            if "final residual =" in line.lower():
                res = float(line.split("final residual =")[1].split(",")[0].strip())
                if res < threshold: converged_u = True
        
        if "Solving for p" in line:
            if "final residual =" in line.lower():
                res = float(line.split("final residual =")[1].split(",")[0].strip())
                if res < threshold: converged_p = True
        
        if converged_u and converged_p:
            return True
            
    return False
