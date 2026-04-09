"""
Module 2.2 — CFD Run Automation

Orchestrates the full geometry → mesh → solve → extract pipeline into a single
callable function for the optimizer.

On failure (mesh error, solver divergence, timeout), returns penalty metrics
instead of crashing the loop.
"""

import logging
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict

from ooc_optimizer.cfd.meshing import generate_mesh
from ooc_optimizer.cfd.metrics import extract_metrics
from ooc_optimizer.geometry import generate_chip

logger = logging.getLogger(__name__)

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


def evaluate_cfd(
    params: Dict[str, float],
    pillar_config: str,
    H_um: float,
    config: dict,
) -> Dict[str, float]:
    """Run the full CFD evaluation pipeline for one parameter set."""
    # Convert height to meters for physics calculations
    H_m = H_um * 1e-6
    work_dir = Path(config["paths"]["work_dir"])
    template_dir = Path(config["paths"]["template_dir"])
    
    # 1. Setup case directory
    case_name = f"run_{pillar_config}_{id(params)}"
    case_dir = work_dir / case_name
    
    try:
        _setup_case(template_dir, case_dir, params, H_m)
        
        # 2. Generate STL and Mesh
        stl_path = case_dir / "constant" / "triSurface" / "domain.stl"
        generate_chip(params, pillar_config, H_um, stl_path)
        
        mesh_path = generate_mesh(stl_path, case_dir, pillar_config, mesh_resolution=1)
        if mesh_path is None:
            return PENALTY_METRICS

        # 3. Run Solver
        if not _run_simplefoam(case_dir):
            return PENALTY_METRICS

        # 4. Extract Metrics
        # Assuming mu is provided in config or params
        mu = config.get("physics", {}).get("mu", 1e-3)
        return extract_metrics(case_dir, H_m, mu)

    except Exception as e:
        logger.error(f"CFD Pipeline failed for {case_name}: {e}")
        return PENALTY_METRICS
    finally:
        # Optional: cleanup large mesh files if needed
        pass


def _setup_case(template_dir: Path, case_dir: Path, params: Dict[str, float], H: float) -> None:
    """Copy template case and update boundary conditions."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    
    # Copy system, constant, and initial fields
    shutil.copytree(template_dir, case_dir)
    
    # Calculate Inlet Velocity: U = Q / (W * H)
    # Q is often in uL/min, convert to m^3/s
    Q_m3s = (params["Q"] * 1e-9) / 60.0
    W_m = params["W"] * 1e-6
    u_inlet = Q_m3s / (W_m * H)

    # Use 'sed' to inject the calculated velocity into the 0/U file
    u_file = case_dir / "0" / "U"
    subprocess.run(
        ["sed", "-i", f"s/INLET_VELOCITY/{u_inlet:.6f}/g", str(u_file)],
        check=True
    )


def _run_simplefoam(case_dir: Path, timeout: int = SOLVER_TIMEOUT_S) -> bool:
    """Execute simpleFoam and return True if converged."""
    log_path = case_dir / "simpleFoam.log"
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                ["simpleFoam"],
                cwd=case_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
                env=os.environ
            )
        return _check_convergence(case_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Solver failed or timed out in {case_dir}")
        return False

def _check_convergence(case_dir: Path, threshold: float = 1e-4) -> bool:
    """Parse solver log for residual convergence."""
    log_path = case_dir / "simpleFoam.log"
    if not log_path.exists():
        return False

    with open(log_path, "r") as f:
        lines = f.readlines()
    
    # Look for the last 'Final residual' entries in the log
    converged_u = False
    converged_p = False
    
    # Search backwards from the end of the log
    for line in reversed(lines):
        if "Solving for Ux" in line or "Solving for Uy" in line:
            parts = line.split(",")
            for part in parts:
                if "final residual" in part.lower():
                    res = float(part.split("=")[1].strip())
                    if res < threshold: converged_u = True
        
        if "Solving for p" in line:
            parts = line.split(",")
            for part in parts:
                if "final residual" in part.lower():
                    res = float(part.split("=")[1].strip())
                    if res < threshold: converged_p = True
        
        if converged_u and converged_p:
            return True
            
    return False
