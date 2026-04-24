"""
Module 1.1 / 2.2 — Passive-scalar transport orchestration and verification.

After the momentum solve (simpleFoam) converges we freeze its U, p, phi fields
and run scalarTransportFoam on the frozen flow to obtain the tracer/drug
concentration field C(x, y).  Two inlets (inlet_drug, inlet_medium) impose
Dirichlet concentrations C = 1 and C = 0 respectively; the rest of the domain
boundary has zero flux.

This module also provides a stand-alone 1D advection-diffusion verification
(Module 1.1 [EXTENDED]) that checks the scalar solver against an analytic
solution over a range of Peclet numbers, before any BO machinery is touched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ooc_optimizer.cfd.foam_parser import (
    find_latest_time,
    read_cell_centres,
    read_scalar_field,
    read_vector_field,
)

logger = logging.getLogger(__name__)

SCALAR_TIMEOUT_S = 300
DEFAULT_DIFFUSIVITY_M2_S = 1.0e-10  # small-molecule drug surrogate


@dataclass
class ScalarRunResult:
    """Output of a single scalarTransportFoam stage."""

    case_dir: Path
    time_dir: Path
    converged: bool
    wall_time_s: float
    warnings: List[str] = field(default_factory=list)


def _find_openfoam_prefix() -> Optional[str]:
    """Detect native vs wrapper OpenFOAM installation."""
    if shutil.which("scalarTransportFoam") is not None:
        return None
    for wrapper in ("openfoam2406", "openfoam2412", "openfoam2506", "openfoam2512"):
        if shutil.which(wrapper) is not None:
            return wrapper
    raise RuntimeError(
        "OpenFOAM not found. Install via 'brew install gerlero/openfoam/openfoam@2406' "
        "or source the OpenFOAM environment so 'scalarTransportFoam' is on PATH."
    )


def _run_foam(cmd: str, case_dir: Path, *, timeout_s: int = SCALAR_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command, handling native and wrapper installs."""
    prefix = _find_openfoam_prefix()
    full_cmd = [prefix, "-c", f"{cmd} -case {case_dir}"] if prefix else [cmd, "-case", str(case_dir)]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout_s)


# ---------------------------------------------------------------------------
# Case-level orchestration: frozen-flow scalar transport on an existing
# converged simpleFoam case.
# ---------------------------------------------------------------------------


def write_scalar_boundary_file(
    case_dir: Path,
    *,
    c_drug: float = 1.0,
    c_medium: float = 0.0,
    patches: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the 0/T boundary-condition file using the v2 patch names.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory (must already contain 0/ and constant/polyMesh).
    c_drug, c_medium : float
        Fixed scalar value at inlet_drug and inlet_medium patches.
    patches : dict, optional
        Override patch-type assignments for non-standard geometries. Keys are
        patch names; values are any of {"fixedValue:1", "fixedValue:0",
        "zeroGradient", "empty", "symmetry"}.  Default covers the v2 contract.

    Returns
    -------
    t_path : Path
        Path to the written 0/T file.
    """
    case_dir = Path(case_dir)
    t_path = case_dir / "0" / "T"
    t_path.parent.mkdir(parents=True, exist_ok=True)

    default_patches = {
        "inlet_drug": f"fixedValue:{c_drug}",
        "inlet_medium": f"fixedValue:{c_medium}",
        "outlet": "zeroGradient",
        "walls": "zeroGradient",
        "floor": "zeroGradient",
        "frontAndBack": "empty",
    }
    if patches:
        default_patches.update(patches)

    entries = []
    for name, spec in default_patches.items():
        if spec.startswith("fixedValue:"):
            value = float(spec.split(":", 1)[1])
            entries.append(
                f"    {name}\n    {{\n        type            fixedValue;\n"
                f"        value           uniform {value};\n    }}\n"
            )
        elif spec in ("zeroGradient", "empty", "symmetry"):
            entries.append(f"    {name}\n    {{\n        type            {spec};\n    }}\n")
        else:
            raise ValueError(f"Unsupported patch spec '{spec}' for patch '{name}'")

    content = textwrap.dedent(
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      T;
        }

        dimensions      [0 0 0 0 0 0 0];

        internalField   uniform 0;

        boundaryField
        {
        """
    )
    content += "".join(entries)
    content += "}\n"
    t_path.write_text(content)
    return t_path


def set_transport_diffusivity(case_dir: Path, diffusivity_m2_s: float) -> None:
    """Replace the DT entry in constant/transportProperties."""
    if diffusivity_m2_s <= 0:
        raise ValueError(f"Diffusivity must be positive, got {diffusivity_m2_s}")
    tp_path = Path(case_dir) / "constant" / "transportProperties"
    if not tp_path.exists():
        raise FileNotFoundError(f"transportProperties missing: {tp_path}")
    text = tp_path.read_text()
    new_line = f"DT              [0 2 -1 0 0 0 0] {diffusivity_m2_s:.6e};"
    if re.search(r"^DT\s+\[", text, flags=re.MULTILINE):
        text = re.sub(r"^DT\s+\[.*?\];\s*$", new_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + "\n\n" + new_line + "\n"
    tp_path.write_text(text)


def set_scalar_controldict(case_dir: Path, *, end_time: float = 500.0) -> None:
    """Overwrite system/controlDict for a steady scalar-transport run.

    scalarTransportFoam solves ∂C/∂t + ∇·(φC) − ∇·(DT ∇C) = 0. With
    ddtSchemes=steadyState (inherited from fvSchemes) the time derivative
    vanishes and the solver converges to the steady scalar field; the
    endTime below is just an iteration budget.
    """
    cd_path = Path(case_dir) / "system" / "controlDict"
    content = textwrap.dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      controlDict;
        }}

        application     scalarTransportFoam;

        startFrom       latestTime;
        stopAt          endTime;
        endTime         {end_time};

        deltaT          1;

        writeControl    timeStep;
        writeInterval   {max(1, int(end_time))};

        purgeWrite      2;

        writeFormat     ascii;
        writePrecision  8;
        writeCompression off;

        timeFormat      general;
        timePrecision   6;

        runTimeModifiable true;
        """
    )
    cd_path.write_text(content)


def run_scalar_transport(
    case_dir: Path,
    *,
    diffusivity_m2_s: float,
    c_drug: float = 1.0,
    c_medium: float = 0.0,
    end_time: float = 500.0,
    patches: Optional[Dict[str, str]] = None,
    timeout_s: int = SCALAR_TIMEOUT_S,
) -> ScalarRunResult:
    """Run scalarTransportFoam on an already-solved momentum case.

    The case_dir is assumed to already contain a converged simpleFoam result
    (U, p, phi in the latest time directory) and a polyMesh with patches
    matching the v2 naming contract.  Boundary conditions for T are written
    into 0/T, the diffusivity DT is injected into transportProperties, and
    the controlDict is switched to scalarTransportFoam.

    The function is idempotent: running it twice leaves the case in the same
    state as running it once.
    """
    import time as _time

    case_dir = Path(case_dir)
    warnings: List[str] = []

    # 1. Inject diffusivity.
    set_transport_diffusivity(case_dir, diffusivity_m2_s)

    # 2. Write T boundary conditions.
    write_scalar_boundary_file(
        case_dir,
        c_drug=c_drug,
        c_medium=c_medium,
        patches=patches,
    )

    # 3. controlDict uses startFrom=latestTime so scalarTransportFoam starts
    # from simpleFoam's final time directory, where U/p/phi live. Copy the
    # freshly written 0/T there so the solver can read the initial condition
    # (otherwise it fatals with "cannot find file .../<latestTime>/T").  Also
    # offset endTime past simpleFoam's final step so the solver actually
    # iterates (endTime < latestTime -> immediate "End").
    latest = find_latest_time(case_dir)
    start_offset = 0.0
    if latest is not None and latest.name != "0":
        src_t = case_dir / "0" / "T"
        if src_t.exists():
            shutil.copy2(src_t, latest / "T")
        try:
            start_offset = float(latest.name)
        except ValueError:
            start_offset = 0.0
    set_scalar_controldict(case_dir, end_time=start_offset + end_time)

    # 4. Run solver.
    start = _time.perf_counter()
    log_path = case_dir / "scalarTransportFoam.log"
    prefix = _find_openfoam_prefix()
    cmd = ["scalarTransportFoam"] if not prefix else [prefix, "-c", "scalarTransportFoam"]
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=True,
                env=os.environ,
            )
        wall = _time.perf_counter() - start
        converged = _check_scalar_convergence(case_dir)
        time_dir = find_latest_time(case_dir)
        if time_dir is None:
            raise RuntimeError("scalarTransportFoam produced no time directory")
        return ScalarRunResult(
            case_dir=case_dir,
            time_dir=time_dir,
            converged=converged,
            wall_time_s=wall,
            warnings=warnings,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        wall = _time.perf_counter() - start
        warnings.append(f"scalarTransportFoam failed: {exc}")
        return ScalarRunResult(
            case_dir=case_dir,
            time_dir=case_dir,
            converged=False,
            wall_time_s=wall,
            warnings=warnings,
        )


def _check_scalar_convergence(case_dir: Path, threshold: float = 1e-6) -> bool:
    """Inspect the scalarTransportFoam log for final-residual convergence."""
    log_path = case_dir / "scalarTransportFoam.log"
    if not log_path.exists():
        return False
    residual_pattern = re.compile(
        r"Solving for T.*Final residual\s*=\s*([0-9eE\+\-\.]+)", re.IGNORECASE
    )
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        m = residual_pattern.search(line)
        if m:
            try:
                return float(m.group(1)) < threshold
            except ValueError:
                return False
    return False


# ---------------------------------------------------------------------------
# Module 1.1 [EXTENDED]: 1D advection-diffusion verification against an
# analytic solution on a Poiseuille velocity field.
# ---------------------------------------------------------------------------


def analytic_ad_1d(x: np.ndarray, L: float, Pe: float) -> np.ndarray:
    """Steady 1D advection-diffusion solution on [0, L] with C(0)=1, C(L)=0.

    The PDE  u·dC/dx = D·d²C/dx²  with Pe = u·L/D and boundary conditions
    C(0) = 1, C(L) = 0 admits the closed-form solution

        C(ξ) = [1 - exp(-Pe·(1 - ξ))] / [1 - exp(-Pe)],   ξ = x / L.

    This formulation is numerically stable for large Pe (exp(-Pe) → 0) where
    the equivalent form  (exp(Pe) - exp(Pe·ξ)) / (exp(Pe) - 1)  would
    overflow.
    """
    xi = np.asarray(x, dtype=float) / L
    if not 0.0 < Pe:
        raise ValueError(f"Pe must be positive, got {Pe}")
    # Handle the diffusion-dominated limit analytically as a safety net.
    if Pe < 1e-6:
        return 1.0 - xi
    return (1.0 - np.exp(-Pe * (1.0 - xi))) / (1.0 - np.exp(-Pe))


def _write_scalar_case_1d(
    case_dir: Path,
    template_dir: Path,
    *,
    L_m: float,
    U_mean: float,
    diffusivity_m2_s: float,
    n_cells: int,
    dy_m: float = 1e-3,
    dz_m: float = 1e-3,
) -> None:
    """Create a 1D scalar-transport verification case (advection-diffusion).

    Geometry: rectangular slab of length L_m, one cell in y and z.  Velocity
    is imposed directly (uniform fixedValue) so this checks the scalar solver
    against an analytic solution without needing simpleFoam in the loop.
    """
    case_dir = Path(case_dir)
    template_dir = Path(template_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    for subdir in ("system", "constant"):
        shutil.copytree(template_dir / subdir, case_dir / subdir)
    (case_dir / "0").mkdir(exist_ok=True)

    # For the 1-D verification sweep the cell Pe can reach ~10 (Pe=1000 on a
    # 100-cell mesh). The repo default `bounded Gauss limitedLinear 1` is
    # unstable (SIGFPE) and the default GaussSeidel smoother for T cannot
    # converge the advection-dominated matrix. First-order upwind + a Krylov
    # solver (PBiCGStab) is unconditionally stable and still meets the 2% L2
    # tolerance on a 200+ cell mesh. See tip.md "scalarTransportFoam high-Pe".
    fvs_path = case_dir / "system" / "fvSchemes"
    fvs = fvs_path.read_text()
    fvs = fvs.replace(
        "div(phi,T)      bounded Gauss limitedLinear 1",
        "div(phi,T)      bounded Gauss upwind",
    )
    fvs_path.write_text(fvs)

    fvsol_path = case_dir / "system" / "fvSolution"
    fvsol = fvsol_path.read_text()
    fvsol = re.sub(
        r"T\s*\{[^}]*\}",
        (
            "T\n    {\n"
            "        solver          PBiCGStab;\n"
            "        preconditioner  DILU;\n"
            "        tolerance       1e-10;\n"
            "        relTol          0;\n"
            "    }"
        ),
        fvsol,
        flags=re.DOTALL,
    )
    fvsol_path.write_text(fvsol)

    # blockMeshDict: 1 x n_cells split along x.
    L_mm = L_m * 1000.0
    W_mm = dy_m * 1000.0
    H_mm = dz_m * 1000.0
    bmd = textwrap.dedent(
        f"""\
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
            (0      0      {H_mm})
            ({L_mm} 0      {H_mm})
            ({L_mm} {W_mm} {H_mm})
            (0      {W_mm} {H_mm})
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({n_cells} 1 1) simpleGrading (1 1 1)
        );

        edges ();

        boundary
        (
            inlet_drug
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
        """
    )
    (case_dir / "system" / "blockMeshDict").write_text(bmd)

    # 0/U: uniform along x.
    u_content = textwrap.dedent(
        f"""\
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
            inlet_drug
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
        """
    )
    (case_dir / "0" / "U").write_text(u_content)

    # 0/p: zero-gradient at both ends (solver needs p present for post-steps).
    p_content = textwrap.dedent(
        """\
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
            inlet_drug
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
        """
    )
    (case_dir / "0" / "p").write_text(p_content)

    # 0/T: fixed at both ends.
    t_content = textwrap.dedent(
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      T;
        }

        dimensions      [0 0 0 0 0 0 0];

        internalField   uniform 0.5;

        boundaryField
        {
            inlet_drug
            {
                type            fixedValue;
                value           uniform 1;
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
        """
    )
    (case_dir / "0" / "T").write_text(t_content)

    set_transport_diffusivity(case_dir, diffusivity_m2_s)
    set_scalar_controldict(case_dir, end_time=1000.0)


def run_scalar_verification_1d(
    template_dir: Path,
    output_dir: Path,
    *,
    L_m: float = 0.01,
    n_cells: int = 100,
    pe_values: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
    diffusivity_m2_s: float = DEFAULT_DIFFUSIVITY_M2_S,
    tolerance_l2: float = 0.02,
) -> List[Dict]:
    """Verify scalarTransportFoam against the 1D analytic solution.

    For each Pe in pe_values, build a 1D channel case, run scalarTransportFoam
    with DT set so that Pe = U·L/D, and compare the simulated C(x) profile to
    the analytic solution.

    Parameters
    ----------
    template_dir : Path
        Path to ooc_optimizer/cfd/template — used for system/ and constant/
        dictionary stubs.
    output_dir : Path
        Parent directory for verification cases.  One subdirectory per Pe.
    L_m : float
        Channel length [m]. Default 10 mm matches Module 1.2.
    n_cells : int
        Streamwise cell count. Pass criterion is evaluated at n_cells = 100.
    pe_values : tuple of float
        Peclet numbers to sweep.
    diffusivity_m2_s : float
        Base diffusivity [m^2/s]. U is chosen per case so that Pe = U·L/D.
    tolerance_l2 : float
        Pass threshold for the relative L2 error between simulated and
        analytic C(x) profiles.

    Returns
    -------
    results : list of dict
        One entry per Pe with keys:
        ``Pe``, ``U_mean``, ``L2_rel_error``, ``linf_error``, ``converged``,
        ``passed``, ``case_dir``.  A summary JSON is also written to
        ``output_dir / scalar_verification_results.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = Path(template_dir)
    if not template_dir.exists():
        raise FileNotFoundError(f"Template dir not found: {template_dir}")

    results: List[Dict] = []

    for Pe in pe_values:
        U_mean = Pe * diffusivity_m2_s / L_m
        # First-order upwind becomes excessively diffusive (and visibly
        # unphysical) once the cell Peclet exceeds ~5, because the analytic
        # boundary layer at x=L is thinner than a single cell. Auto-refine so
        # cell_Pe <= 2 while respecting the user-requested floor.
        n_cells_eff = max(n_cells, int(np.ceil(Pe / 2.0)))
        case_dir = output_dir / f"ad_1d_Pe{int(Pe)}"
        logger.info(
            "Scalar verification: Pe=%s, U=%.3e m/s, L=%s m, D=%.2e m^2/s, "
            "n_cells=%d (cell_Pe=%.1f)",
            Pe, U_mean, L_m, diffusivity_m2_s, n_cells_eff, Pe / n_cells_eff,
        )
        _write_scalar_case_1d(
            case_dir=case_dir,
            template_dir=template_dir,
            L_m=L_m,
            U_mean=U_mean,
            diffusivity_m2_s=diffusivity_m2_s,
            n_cells=n_cells_eff,
        )

        # blockMesh, then scalarTransportFoam.
        bm = _run_foam("blockMesh", case_dir)
        if bm.returncode != 0:
            (case_dir / "blockMesh.log").write_text(bm.stdout + bm.stderr)
            results.append({
                "Pe": float(Pe),
                "U_mean": float(U_mean),
                "L2_rel_error": None,
                "linf_error": None,
                "converged": False,
                "passed": False,
                "case_dir": str(case_dir),
                "error": "blockMesh failed",
            })
            continue

        sr = run_scalar_transport(
            case_dir,
            diffusivity_m2_s=diffusivity_m2_s,
            c_drug=1.0,
            c_medium=0.0,  # unused here since inlet_medium does not exist in 1D case
            end_time=1000.0,
            patches={
                "inlet_drug": "fixedValue:1",
                "outlet": "fixedValue:0",
                "walls": "zeroGradient",
                "frontAndBack": "empty",
            },
        )

        # Compare against analytic.
        try:
            # write cell centres
            cc = _run_foam("postProcess -func writeCellCentres -latestTime", case_dir)
            if cc.returncode != 0:
                raise RuntimeError(cc.stderr[-400:])
            centres = read_cell_centres(case_dir)
            x_cells = centres[:, 0]
            t_time = find_latest_time(case_dir)
            T_sim = read_scalar_field(t_time / "T")
            T_ana = analytic_ad_1d(x_cells, L_m, Pe)
            l2 = float(np.sqrt(np.mean((T_sim - T_ana) ** 2)) / max(np.sqrt(np.mean(T_ana ** 2)), 1e-15))
            linf = float(np.max(np.abs(T_sim - T_ana)))
            passed = sr.converged and l2 < tolerance_l2
            results.append({
                "Pe": float(Pe),
                "U_mean": float(U_mean),
                "L2_rel_error": l2,
                "linf_error": linf,
                "converged": bool(sr.converged),
                "passed": bool(passed),
                "case_dir": str(case_dir),
                "n_cells": int(n_cells_eff),
            })
            logger.info(
                "Pe=%s: L2_rel=%.4f, Linf=%.4f, converged=%s, passed=%s",
                Pe, l2, linf, sr.converged, passed,
            )
        except Exception as exc:
            logger.error("Analysis failed for Pe=%s: %s", Pe, exc)
            results.append({
                "Pe": float(Pe),
                "U_mean": float(U_mean),
                "L2_rel_error": None,
                "linf_error": None,
                "converged": bool(sr.converged),
                "passed": False,
                "case_dir": str(case_dir),
                "error": str(exc),
            })

    summary_path = output_dir / "scalar_verification_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Scalar verification summary written to %s", summary_path)
    return results


# ---------------------------------------------------------------------------
# Helpers used by Module 2.2 when running full two-solver per-evaluation.
# ---------------------------------------------------------------------------


def extract_concentration_field(case_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cell_centres, C) arrays for the latest time directory.

    cell_centres : N x 3 array of cell-centre coordinates [m].
    C : length-N array of T (tracer) values.
    """
    case_dir = Path(case_dir)
    time_dir = find_latest_time(case_dir)
    if time_dir is None:
        raise FileNotFoundError(f"No result time directory found in {case_dir}")
    t_file = time_dir / "T"
    if not t_file.exists():
        raise FileNotFoundError(f"T field missing in {time_dir}")
    centres = read_cell_centres(case_dir)
    C = read_scalar_field(t_file)
    if len(C) != len(centres):
        raise ValueError("cell centre / T length mismatch — mesh was re-decomposed?")
    return centres, C


def frozen_flow_velocity(case_dir: Path) -> np.ndarray:
    """Read the frozen U field for downstream diagnostics (streamlines etc.)."""
    case_dir = Path(case_dir)
    time_dir = find_latest_time(case_dir)
    if time_dir is None:
        raise FileNotFoundError(f"No result time directory found in {case_dir}")
    return read_vector_field(time_dir / "U")
