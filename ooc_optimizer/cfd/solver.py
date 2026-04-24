"""
Module 2.2 — CFD Run Automation (v2).

Per evaluation:
    1. Stage an OpenFOAM case with topology-aware blockMeshDict (v2 patches).
    2. Run blockMesh (+ snappyHexMesh if pillars).
    3. Run simpleFoam on the momentum BCs (U for two inlets, split by r_flow).
    4. Run scalarTransportFoam on the frozen U, with C = 1 on inlet_drug and
       C = 0 on inlet_medium, producing a concentration field.
    5. Extract the v2 metrics dict: L2_to_target, grad_sharpness,
       monotonicity, f_dead, tau_mean, plus convergence flags.

Failure modes return the standard penalty dict so that the BO loop can
apply a penalty objective without crashing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ooc_optimizer.cfd.meshing import MeshResult, generate_mesh
from ooc_optimizer.cfd.metrics import extract_v2_metrics
from ooc_optimizer.cfd.scalar import (
    DEFAULT_DIFFUSIVITY_M2_S,
    _find_openfoam_prefix,
    run_scalar_transport,
)
from ooc_optimizer.geometry import generate_blockmesh_dict_v2, generate_pillar_obstacles_stl
from ooc_optimizer.optimization.objectives import TargetProfile

logger = logging.getLogger(__name__)

SOLVER_TIMEOUT_S = 300

PENALTY_METRICS: Dict[str, Any] = {
    "L2_to_target": 99.0,
    "grad_sharpness": 0.0,
    "monotonicity": 0.0,
    "tau_mean": 0.0,
    "tau_min": 0.0,
    "tau_max": 0.0,
    "f_dead": 1.0,
    "delta_p": 0.0,
    "cv_tau": 999.0,  # retained for backward compatibility / diagnostics
    "converged_U": False,
    "converged_C": False,
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


def evaluate_cfd(
    params: Dict[str, float],
    pillar_config: str,
    H_um: float,
    config: dict,
    *,
    topology: str = "opposing",
    target_profile: Optional[TargetProfile] = None,
) -> Dict[str, Any]:
    """Run one full (momentum + scalar) CFD evaluation.

    Parameters
    ----------
    params : dict
        Continuous parameters. Keys depend on topology; see
        ``ooc_optimizer.geometry.generator`` for the contract.
    pillar_config : str
        Pillar layout.
    H_um : float
        Chamber height [μm].
    config : dict
        Loaded YAML config; must supply ``paths``, ``fixed_parameters``,
        ``solver_settings`` and ``diffusivity`` (top-level float).
    topology : str
        v2 inlet topology.
    target_profile : TargetProfile, optional
        Profile used to compute the primary objective ``L2_to_target``.
        When None, L2_to_target is left as NaN (the BO loop should always
        supply a target).

    Returns
    -------
    metrics : dict
        See the PENALTY_METRICS keys above for the contract.
    """
    H_m = H_um * 1e-6
    work_dir = Path(config["paths"]["work_dir"])
    template_dir = Path(config["paths"]["template_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    case_name = f"run_{topology}_{pillar_config}_H{int(H_um)}_{int(time.time()*1000)}"
    case_dir = work_dir / case_name

    try:
        bm = _setup_case(template_dir, case_dir, params, topology, H_m, pillar_config)

        # Pillar STL used by snappyHexMesh.
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
            return _penalty(case_dir, warnings=["Mesh generation failed before solver execution."])

        # ----- momentum solve -----------------------------------------------
        if not _run_simplefoam(case_dir):
            return _penalty(
                case_dir,
                mesh_result=mesh_result,
                warnings=["simpleFoam failed to converge."],
            )

        # Write cell centres for metric extraction / target evaluation.
        _run_postprocess(case_dir, "writeCellCentres")

        # ----- passive-scalar solve ----------------------------------------
        diffusivity = float(config.get("diffusivity", DEFAULT_DIFFUSIVITY_M2_S))
        scalar_result = run_scalar_transport(
            case_dir=case_dir,
            diffusivity_m2_s=diffusivity,
            c_drug=1.0,
            c_medium=0.0,
        )
        converged_C = scalar_result.converged

        # ----- metric extraction -------------------------------------------
        mu = float(config["fixed_parameters"]["fluid_viscosity_Pa_s"])
        metrics = extract_v2_metrics(
            case_dir=case_dir,
            H=H_m,
            mu=mu,
            chamber_length_m=bm.chamber_length_m,
            chamber_width_m=bm.chamber_width_m,
            target_profile=target_profile,
        )
        metrics["converged_U"] = True
        metrics["converged_C"] = bool(converged_C)
        metrics["converged"] = bool(metrics.get("converged") and converged_C)
        metrics["mesh_ok"] = mesh_result.mesh_ok
        metrics["mesh_strategy_requested"] = mesh_result.strategy_requested
        metrics["mesh_strategy_used"] = mesh_result.strategy_used
        metrics["mesh_checkmesh_ok"] = mesh_result.checkmesh_ok
        metrics["mesh_used_fallback"] = mesh_result.used_fallback
        metrics["mesh_checkmesh_summary"] = dict(mesh_result.checkmesh_summary)
        metrics["warnings"] = list(mesh_result.warnings) + list(scalar_result.warnings)
        metrics["case_dir"] = str(case_dir)
        metrics["topology"] = topology
        return metrics

    except Exception as exc:
        logger.error("CFD pipeline failed for %s: %s", case_name, exc, exc_info=True)
        return _penalty(case_dir, warnings=[f"CFD pipeline exception: {exc}"])


def _penalty(case_dir: Path, *, mesh_result: Optional[MeshResult] = None, warnings=None) -> Dict[str, Any]:
    failure = dict(PENALTY_METRICS)
    failure["case_dir"] = str(case_dir)
    failure["warnings"] = list(warnings or [])
    if mesh_result is not None:
        failure["mesh_ok"] = mesh_result.mesh_ok
        failure["mesh_strategy_requested"] = mesh_result.strategy_requested
        failure["mesh_strategy_used"] = mesh_result.strategy_used
        failure["mesh_checkmesh_ok"] = mesh_result.checkmesh_ok
        failure["mesh_used_fallback"] = mesh_result.used_fallback
        failure["mesh_checkmesh_summary"] = dict(mesh_result.checkmesh_summary)
        failure["warnings"] = list(mesh_result.warnings) + failure["warnings"]
    return failure


def _setup_case(
    template_dir: Path,
    case_dir: Path,
    params: Dict[str, float],
    topology: str,
    H_m: float,
    pillar_config: str,
):
    """Copy template and inject topology-aware blockMeshDict + two-inlet BCs."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)

    # -- blockMeshDict -------------------------------------------------------
    bm = generate_blockmesh_dict_v2(
        params=params,
        topology=topology,
        H_um=H_m * 1e6,
        dz_mm=0.01,
    )
    (case_dir / "system" / "blockMeshDict").write_text(bm.content)

    # -- momentum BCs --------------------------------------------------------
    Q_total_m3s = float(params["Q_total"]) * 1e-9 / 60.0
    r_flow = float(params["r_flow"])
    Q_drug = r_flow * Q_total_m3s
    Q_medium = (1.0 - r_flow) * Q_total_m3s
    U_drug = Q_drug / max(bm.inlet_drug_area_m2, 1e-20)
    U_medium = Q_medium / max(bm.inlet_medium_area_m2, 1e-20)

    # For asymmetric_lumen the drug inlet is on the y=0 wall so the inflow
    # direction is +y; all other topologies use +x at x=0.
    if topology == "asymmetric_lumen":
        drug_vec = (0.0, U_drug, 0.0)
    else:
        drug_vec = (U_drug, 0.0, 0.0)
    medium_vec = (U_medium, 0.0, 0.0)

    _write_u_field(case_dir / "0" / "U", drug_vec, medium_vec, pillar_config=pillar_config)
    _write_p_field(case_dir / "0" / "p", pillar_config=pillar_config)
    _rewrite_scalar_template(case_dir / "0" / "T", pillar_config=pillar_config)
    return bm


def _write_u_field(u_path: Path, drug_vec, medium_vec, *, pillar_config: str) -> None:
    """Emit a 0/U file with two fixedValue inlet patches + noSlip everywhere else."""
    pillar_block = ""
    if pillar_config.lower() != "none":
        pillar_block = "    pillars\n    {\n        type            noSlip;\n    }\n"
    content = (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       volVectorField;\n"
        "    object      U;\n"
        "}\n\n"
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        f"    inlet_drug\n    {{\n        type            fixedValue;\n"
        f"        value           uniform ({drug_vec[0]:.6f} {drug_vec[1]:.6f} {drug_vec[2]:.6f});\n    }}\n\n"
        f"    inlet_medium\n    {{\n        type            fixedValue;\n"
        f"        value           uniform ({medium_vec[0]:.6f} {medium_vec[1]:.6f} {medium_vec[2]:.6f});\n    }}\n\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n\n"
        "    walls\n    {\n        type            noSlip;\n    }\n\n"
        "    floor\n    {\n        type            noSlip;\n    }\n\n"
        f"{pillar_block}"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n"
    )
    if pillar_config.lower() != "none":
        content = content.replace("type            empty;", "type            symmetry;")
    u_path.write_text(content)


def _write_p_field(p_path: Path, *, pillar_config: str) -> None:
    pillar_block = ""
    if pillar_config.lower() != "none":
        pillar_block = "    pillars\n    {\n        type            zeroGradient;\n    }\n"
    content = (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       volScalarField;\n"
        "    object      p;\n"
        "}\n\n"
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet_drug\n    {\n        type            zeroGradient;\n    }\n\n"
        "    inlet_medium\n    {\n        type            zeroGradient;\n    }\n\n"
        "    outlet\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n\n"
        "    walls\n    {\n        type            zeroGradient;\n    }\n\n"
        "    floor\n    {\n        type            zeroGradient;\n    }\n\n"
        f"{pillar_block}"
        "    frontAndBack\n    {\n        type            empty;\n    }\n"
        "}\n"
    )
    if pillar_config.lower() != "none":
        content = content.replace("type            empty;", "type            symmetry;")
    p_path.write_text(content)


def _rewrite_scalar_template(t_path: Path, *, pillar_config: str) -> None:
    """Ensure the 0/T template carries the correct empty/symmetry for frontAndBack."""
    if not t_path.exists():
        return
    text = t_path.read_text()
    if pillar_config.lower() != "none":
        text = text.replace("type            empty;", "type            symmetry;")
    # Always rewrite to pick up pillar patch when required.
    if pillar_config.lower() != "none" and "pillars" not in text:
        text = text.replace(
            "    frontAndBack",
            "    pillars\n    {\n        type            zeroGradient;\n    }\n\n    frontAndBack",
        )
    t_path.write_text(text)


def _run_simplefoam(case_dir: Path, timeout: int = SOLVER_TIMEOUT_S) -> bool:
    log_path = case_dir / "simpleFoam.log"
    prefix = _find_openfoam_prefix()
    cmd = ["simpleFoam"] if not prefix else [prefix, "-c", "simpleFoam"]
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
                env=os.environ,
            )
        return _check_momentum_convergence(case_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _run_postprocess(case_dir: Path, func: str) -> None:
    prefix = _find_openfoam_prefix()
    cmd = ["postProcess", "-func", func, "-latestTime"]
    if prefix:
        cmd = [prefix, "-c", " ".join(cmd)]
    subprocess.run(cmd, cwd=case_dir, check=True, capture_output=True, env=os.environ)


def _check_momentum_convergence(case_dir: Path, threshold: float = 1e-4) -> bool:
    log_path = case_dir / "simpleFoam.log"
    if not log_path.exists():
        return False
    with open(log_path, "r") as f:
        lines = f.readlines()
    residual_pattern = re.compile(r"final residual\s*=\s*([0-9eE\+\-\.]+)", re.IGNORECASE)
    converged_u, converged_p = False, False
    for line in reversed(lines):
        if any(v in line for v in ("Solving for Ux", "Solving for Uy")):
            m = residual_pattern.search(line)
            if m and float(m.group(1)) < threshold:
                converged_u = True
        if "Solving for p" in line:
            m = residual_pattern.search(line)
            if m and float(m.group(1)) < threshold:
                converged_p = True
        if converged_u and converged_p:
            return True
    return False
