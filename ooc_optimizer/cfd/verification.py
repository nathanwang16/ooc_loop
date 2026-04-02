"""
Module 1.1 — Poiseuille Flow Verification

Analytical solutions for 2D Poiseuille flow and automated verification
against OpenFOAM simulation results.

Verification geometry: straight rectangular channel with known dimensions.
The 2D plan-view simulation is one cell thick in z with empty BCs.
Analytical centerline velocity and floor WSS are compared to simulated values.

Agreement within 2% confirms solver, mesh resolution, and post-processing.
"""

import json
import logging
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ooc_optimizer.cfd.foam_parser import (
    find_latest_time,
    read_cell_centres,
    read_vector_field,
)

logger = logging.getLogger(__name__)


@dataclass
class PoiseuilleSolution:
    """Analytical 2D Poiseuille channel flow solution.

    Geometry:
        x ∈ [0, L]  — streamwise (flow direction)
        y ∈ [0, W]  — cross-stream (between side walls)
        H           — channel depth (for floor WSS estimation)

    All units SI (m, Pa·s, kg/m³, m/s).
    """

    L: float      # channel length [m]
    W: float      # channel width [m]
    H: float      # channel height [m]
    Q_ul_min: float  # volumetric flow rate [μL/min]
    mu: float     # dynamic viscosity [Pa·s]
    rho: float    # density [kg/m³]

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m²/s]."""
        return self.mu / self.rho

    @property
    def Q_m3s(self) -> float:
        """Volumetric flow rate [m³/s]."""
        return self.Q_ul_min * 1e-9 / 60.0

    @property
    def U_mean(self) -> float:
        """Mean (bulk) velocity through the cross-section [m/s]."""
        return self.Q_m3s / (self.W * self.H)

    @property
    def U_centerline(self) -> float:
        """Centerline velocity for 2D Poiseuille flow [m/s].

        u_max = (3/2) × U_mean for flow between parallel plates.
        """
        return 1.5 * self.U_mean

    @property
    def Re(self) -> float:
        """Reynolds number based on channel width."""
        return self.U_mean * self.W / self.nu

    @property
    def development_length(self) -> float:
        """Hydrodynamic entrance length [m].

        L_dev ≈ 0.05 × Re × W for channel flow.
        """
        return 0.05 * self.Re * self.W

    @property
    def dp_dx(self) -> float:
        """Streamwise pressure gradient [Pa/m] (negative = driving flow)."""
        return -12.0 * self.mu * self.U_mean / (self.W ** 2)

    @property
    def pressure_drop_kinematic(self) -> float:
        """Kinematic pressure drop across the channel [m²/s²].

        Δ(p/ρ) = 12 ν L U_mean / W²
        """
        return 12.0 * self.nu * self.L * self.U_mean / (self.W ** 2)

    @property
    def pressure_drop_Pa(self) -> float:
        """Dynamic pressure drop [Pa]."""
        return self.pressure_drop_kinematic * self.rho

    def velocity_profile(self, y: np.ndarray) -> np.ndarray:
        """Analytical velocity u_x(y) for fully-developed flow.

        u(y) = (6 U_mean / W²) × y × (W − y)

        Parameters
        ----------
        y : array-like
            Cross-stream positions [m], must be in [0, W].

        Returns
        -------
        u_x : ndarray
            Streamwise velocity [m/s].
        """
        y = np.asarray(y, dtype=float)
        return (6.0 * self.U_mean / self.W**2) * y * (self.W - y)

    def floor_wss(self, y: np.ndarray) -> np.ndarray:
        """Estimated floor WSS from the 2D depth-averaged velocity.

        τ_floor(y) = 6 μ u(y) / H

        Parameters
        ----------
        y : array-like
            Cross-stream positions [m].

        Returns
        -------
        tau_floor : ndarray
            Floor wall shear stress [Pa].
        """
        u = self.velocity_profile(y)
        return 6.0 * self.mu * u / self.H

    @property
    def floor_wss_mean(self) -> float:
        """Area-averaged floor WSS [Pa].

        τ_floor_mean = 6 μ U_mean / H
        """
        return 6.0 * self.mu * self.U_mean / self.H

    @property
    def floor_wss_centerline(self) -> float:
        """Floor WSS at channel centerline [Pa].

        τ_floor(W/2) = 6 μ U_centerline / H = 9 μ U_mean / H
        """
        return 6.0 * self.mu * self.U_centerline / self.H


def generate_blockmesh_dict(
    L_mm: float,
    W_mm: float,
    dz_mm: float,
    nx: int,
    ny: int,
) -> str:
    """Generate a blockMeshDict for a straight rectangular 2D channel.

    Parameters
    ----------
    L_mm : float
        Channel length in mm (x-direction).
    W_mm : float
        Channel width in mm (y-direction).
    dz_mm : float
        Slab thickness in mm (z-direction, 1 cell).
    nx : int
        Number of cells along the length.
    ny : int
        Number of cells across the width.

    Returns
    -------
    content : str
        Complete blockMeshDict file content.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Cell counts must be positive: nx={nx}, ny={ny}")
    if L_mm <= 0 or W_mm <= 0 or dz_mm <= 0:
        raise ValueError(f"Dimensions must be positive: L={L_mm}, W={W_mm}, dz={dz_mm}")

    return textwrap.dedent(f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      blockMeshDict;
        }}

        convertToMeters 0.001;

        vertices
        (
            (0      0      0)
            ({L_mm} 0      0)
            ({L_mm} {W_mm} 0)
            (0      {W_mm} 0)
            (0      0      {dz_mm})
            ({L_mm} 0      {dz_mm})
            ({L_mm} {W_mm} {dz_mm})
            (0      {W_mm} {dz_mm})
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
        );

        edges ();

        boundary
        (
            inlet
            {{
                type patch;
                faces ((0 4 7 3));
            }}
            outlet
            {{
                type patch;
                faces ((2 6 5 1));
            }}
            walls
            {{
                type wall;
                faces (
                    (1 5 4 0)
                    (3 7 6 2)
                );
            }}
            frontAndBack
            {{
                type empty;
                faces (
                    (0 3 2 1)
                    (4 5 6 7)
                );
            }}
        );

        mergePatchPairs ();
    """)


def generate_inlet_U(U_mean: float, W_m: float) -> str:
    """Generate the 0/U file with uniform inlet velocity.

    Uses a uniform inlet rather than a parabolic profile.  The flow develops
    a parabolic profile within L_dev ≈ 0.05·Re·W, which is <1 mm for our
    Re range.  Verification compares only in the fully-developed region.

    Parameters
    ----------
    U_mean : float
        Mean inlet velocity [m/s].
    W_m : float
        Channel width [m] (kept in signature for consistency; not used for
        uniform inlet but needed when parabolic inlets are re-enabled).

    Returns
    -------
    content : str
        Complete 0/U file content.
    """
    return textwrap.dedent(f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       volVectorField;
            object      U;
        }}

        dimensions      [0 1 -1 0 0 0 0];

        internalField   uniform ({U_mean} 0 0);

        boundaryField
        {{
            inlet
            {{
                type            fixedValue;
                value           uniform ({U_mean} 0 0);
            }}

            outlet
            {{
                type            zeroGradient;
            }}

            walls
            {{
                type            noSlip;
            }}

            frontAndBack
            {{
                type            empty;
            }}
        }}
    """)


def generate_p_file() -> str:
    """Generate the 0/p file for the verification case."""
    return textwrap.dedent("""\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      p;
        }

        dimensions      [0 2 -2 0 0 0 0];

        internalField   uniform 0;

        boundaryField
        {
            inlet
            {
                type            zeroGradient;
            }

            outlet
            {
                type            fixedValue;
                value           uniform 0;
            }

            walls
            {
                type            zeroGradient;
            }

            frontAndBack
            {
                type            empty;
            }
        }
    """)


def _find_openfoam_prefix() -> Optional[str]:
    """Detect the OpenFOAM invocation method.

    Returns the wrapper command (e.g. 'openfoam2406') if OpenFOAM is installed
    via the macOS .app bundle, or None if native commands are on PATH.

    Raises RuntimeError if no OpenFOAM installation is found.
    """
    if shutil.which("simpleFoam") is not None:
        return None

    for wrapper in ("openfoam2406", "openfoam2412", "openfoam2506", "openfoam2512"):
        if shutil.which(wrapper) is not None:
            return wrapper

    raise RuntimeError(
        "OpenFOAM not found. Install via 'brew install gerlero/openfoam/openfoam@2406' "
        "or source your OpenFOAM environment so that 'simpleFoam' is on PATH."
    )


def _run_foam_command(
    cmd: str,
    case_dir: Path,
    timeout_s: int = 300,
) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command, handling both native and wrapper installs."""
    prefix = _find_openfoam_prefix()

    if prefix is not None:
        full_cmd = [prefix, "-c", f"{cmd} -case {case_dir}"]
    else:
        full_cmd = [cmd, "-case", str(case_dir)]

    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def setup_verification_case(
    case_dir: Path,
    template_dir: Path,
    sol: PoiseuilleSolution,
    nx: int,
    ny: int,
    dz_mm: float = 0.01,
) -> Path:
    """Create a complete OpenFOAM case for Poiseuille verification.

    Copies the template (system/, constant/), then writes a blockMeshDict
    and boundary-condition files specific to the verification geometry.

    Parameters
    ----------
    case_dir : Path
        Directory to create for this case.
    template_dir : Path
        Path to the OpenFOAM template case.
    sol : PoiseuilleSolution
        Analytical solution instance defining the geometry/flow.
    nx, ny : int
        Mesh cell counts (streamwise, cross-stream).
    dz_mm : float
        Slab thickness in mm.

    Returns
    -------
    case_dir : Path
    """
    case_dir = Path(case_dir)
    template_dir = Path(template_dir)

    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_dir}")

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    for subdir in ("system", "constant"):
        shutil.copytree(template_dir / subdir, case_dir / subdir)

    (case_dir / "0").mkdir(exist_ok=True)

    L_mm = sol.L * 1000.0
    W_mm = sol.W * 1000.0

    bmd = generate_blockmesh_dict(L_mm, W_mm, dz_mm, nx, ny)
    (case_dir / "system" / "blockMeshDict").write_text(bmd)

    u_content = generate_inlet_U(sol.U_mean, sol.W)
    (case_dir / "0" / "U").write_text(u_content)

    p_content = generate_p_file()
    (case_dir / "0" / "p").write_text(p_content)

    logger.info(
        "Verification case set up: %s (nx=%d, ny=%d, U_mean=%.4e m/s)",
        case_dir, nx, ny, sol.U_mean,
    )
    return case_dir


def run_openfoam_case(case_dir: Path, timeout_s: int = 300) -> bool:
    """Execute blockMesh + simpleFoam in the given case directory.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory.
    timeout_s : int
        Maximum wall-clock time for the solver.

    Returns
    -------
    converged : bool
        True if simpleFoam ran to completion without error.
    """
    case_dir = Path(case_dir)

    log_dir = case_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    for step in ("blockMesh", "simpleFoam"):
        log_file = log_dir / f"{step}.log"
        logger.info("Running %s in %s ...", step, case_dir.name)
        try:
            result = _run_foam_command(step, case_dir, timeout_s=timeout_s)
            log_file.write_text(result.stdout + "\n" + result.stderr)

            if result.returncode != 0:
                logger.error(
                    "%s failed (exit %d) in %s. See %s",
                    step, result.returncode, case_dir.name, log_file,
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("%s timed out after %ds in %s", step, timeout_s, case_dir.name)
            return False

    logger.info("Case %s converged successfully", case_dir.name)
    return True


def write_cell_centres(case_dir: Path) -> None:
    """Run 'postProcess -func writeCellCentres' to generate cell centre data."""
    prefix = _find_openfoam_prefix()

    if prefix is not None:
        full_cmd = [prefix, "-c", f"postProcess -func writeCellCentres -case {case_dir}"]
    else:
        full_cmd = ["postProcess", "-func", "writeCellCentres", "-case", str(case_dir)]

    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"writeCellCentres failed in {case_dir}: {result.stderr[-500:]}"
        )


def extract_verification_results(
    case_dir: Path,
    sol: PoiseuilleSolution,
) -> Dict:
    """Compare simulated results to analytical Poiseuille solution.

    Parameters
    ----------
    case_dir : Path
        Converged OpenFOAM case directory.
    sol : PoiseuilleSolution
        Analytical solution for comparison.

    Returns
    -------
    results : dict
        Comparison metrics including errors and per-cell data.
    """
    case_dir = Path(case_dir)

    write_cell_centres(case_dir)

    centres = read_cell_centres(case_dir)
    time_dir = find_latest_time(case_dir)
    if time_dir is None:
        raise FileNotFoundError(f"No result time directory found in {case_dir}")

    U_sim = read_vector_field(time_dir / "U")
    Ux_sim = U_sim[:, 0]

    y_cells = centres[:, 1]
    x_cells = centres[:, 0]

    developed_mask = (x_cells > sol.development_length * 3) & (
        x_cells < sol.L - sol.development_length * 3
    )

    Ux_analytical = sol.velocity_profile(y_cells)

    Ux_sim_dev = Ux_sim[developed_mask]
    Ux_ana_dev = Ux_analytical[developed_mask]
    y_dev = y_cells[developed_mask]

    velocity_errors = np.abs(Ux_sim_dev - Ux_ana_dev) / np.maximum(Ux_ana_dev, 1e-15)

    dist_to_center = np.abs(y_dev - sol.W / 2)
    min_dist = dist_to_center.min()
    centerline_mask = dist_to_center <= min_dist * 1.01 + 1e-15
    if centerline_mask.sum() == 0:
        raise ValueError("No cells found near the centerline for comparison")

    U_cl_sim = Ux_sim_dev[centerline_mask].mean()
    U_cl_ana = sol.U_centerline
    centerline_error = abs(U_cl_sim - U_cl_ana) / U_cl_ana

    tau_sim = 6.0 * sol.mu * Ux_sim_dev / sol.H
    tau_ana = sol.floor_wss(y_dev)
    wss_errors = np.abs(tau_sim - tau_ana) / np.maximum(tau_ana, 1e-15)

    tau_mean_sim = tau_sim.mean()
    tau_mean_ana = sol.floor_wss_mean
    tau_mean_error = abs(tau_mean_sim - tau_mean_ana) / tau_mean_ana

    results = {
        "U_centerline_sim": float(U_cl_sim),
        "U_centerline_ana": float(U_cl_ana),
        "centerline_velocity_error": float(centerline_error),
        "tau_mean_sim": float(tau_mean_sim),
        "tau_mean_ana": float(tau_mean_ana),
        "tau_mean_error": float(tau_mean_error),
        "velocity_max_error": float(velocity_errors.max()),
        "velocity_mean_error": float(velocity_errors.mean()),
        "wss_max_error": float(wss_errors.max()),
        "wss_mean_error": float(wss_errors.mean()),
        "n_cells_developed": int(developed_mask.sum()),
        "n_cells_total": int(len(Ux_sim)),
        "passed_2pct": bool(centerline_error < 0.02 and tau_mean_error < 0.02),
    }

    logger.info(
        "Verification: U_cl error=%.4f%%, τ_mean error=%.4f%%, PASS=%s",
        centerline_error * 100,
        tau_mean_error * 100,
        results["passed_2pct"],
    )
    return results


def run_mesh_convergence(
    template_dir: Path,
    output_dir: Path,
    sol: PoiseuilleSolution,
    refinement_levels: Optional[List[int]] = None,
) -> List[Dict]:
    """Run mesh convergence study at multiple refinement levels.

    Parameters
    ----------
    template_dir : Path
        Path to the OpenFOAM template case.
    output_dir : Path
        Parent directory for convergence study cases.
    sol : PoiseuilleSolution
        Analytical solution.
    refinement_levels : list of int, optional
        Multipliers for the base mesh. Default: [1, 2, 4].

    Returns
    -------
    convergence_data : list of dict
        One entry per refinement level with mesh size and errors.
    """
    if refinement_levels is None:
        refinement_levels = [1, 2, 4]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_nx, base_ny = 100, 10
    convergence_data = []

    for level in refinement_levels:
        nx = base_nx * level
        ny = base_ny * level
        n_cells = nx * ny

        case_name = f"convergence_{level}x"
        case_dir = output_dir / case_name

        logger.info(
            "Convergence study %dx: nx=%d, ny=%d, ~%d cells",
            level, nx, ny, n_cells,
        )

        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_dir,
            sol=sol,
            nx=nx,
            ny=ny,
        )

        converged = run_openfoam_case(case_dir)
        if not converged:
            logger.error("Convergence study %dx failed to converge", level)
            convergence_data.append({
                "level": level,
                "nx": nx,
                "ny": ny,
                "n_cells": n_cells,
                "converged": False,
            })
            continue

        results = extract_verification_results(case_dir, sol)
        results.update({
            "level": level,
            "nx": nx,
            "ny": ny,
            "n_cells": n_cells,
            "converged": True,
        })
        convergence_data.append(results)

    if len(convergence_data) >= 2:
        converged_results = [r for r in convergence_data if r.get("converged", False)]
        if len(converged_results) >= 2:
            last_two = converged_results[-2:]
            tau_change = abs(
                last_two[1]["tau_mean_sim"] - last_two[0]["tau_mean_sim"]
            ) / max(abs(last_two[0]["tau_mean_sim"]), 1e-15)

            logger.info(
                "Mesh convergence: τ_mean change between %dx and %dx = %.4f%%",
                last_two[0]["level"],
                last_two[1]["level"],
                tau_change * 100,
            )
            convergence_data[-1]["tau_mean_relative_change"] = float(tau_change)

    summary_path = output_dir / "convergence_results.json"
    with open(summary_path, "w") as f:
        json.dump(convergence_data, f, indent=2)
    logger.info("Convergence results written to %s", summary_path)

    return convergence_data
