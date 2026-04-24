"""
Minimal 3D sanity validation utilities for the premise test.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ooc_optimizer.cfd.foam_parser import find_latest_time, read_cell_centres, read_vector_field
from ooc_optimizer.cfd.verification import _find_openfoam_prefix

logger = logging.getLogger(__name__)


def run_3d_matched_rectangle(
    W_um: float,
    H_um: float,
    L_um: float,
    Q_ul_min: float,
    mu: float,
    rho: float,
    work_dir: Path,
    nx: int = 300,
    ny: int = 75,
    nz: int = 25,
    residual_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run one 3D rectangular matched-inlet sanity case."""
    case_dir = Path(work_dir) / f"matched_3d_W{int(W_um)}_H{int(H_um)}_Q{int(Q_ul_min)}"
    _prepare_3d_case(case_dir=case_dir)

    W_m = W_um * 1e-6
    H_m = H_um * 1e-6
    L_m = L_um * 1e-6
    Q_m3s = Q_ul_min * 1e-9 / 60.0
    u_inlet = Q_m3s / (W_m * H_m)
    nu = mu / rho

    (case_dir / "system" / "blockMeshDict").write_text(
        _generate_3d_blockmesh_dict(L_m=L_m, W_m=W_m, H_m=H_m, nx=nx, ny=ny, nz=nz),
        encoding="utf-8",
    )
    (case_dir / "0" / "U").write_text(_generate_3d_u_file(u_inlet=u_inlet), encoding="utf-8")
    (case_dir / "0" / "p").write_text(_generate_3d_p_file(), encoding="utf-8")
    (case_dir / "constant" / "transportProperties").write_text(
        _generate_transport_properties(nu=nu), encoding="utf-8"
    )
    (case_dir / "constant" / "turbulenceProperties").write_text(
        _generate_turbulence_properties(), encoding="utf-8"
    )
    (case_dir / "system" / "fvSolution").write_text(
        _generate_fvsolution(residual_tol=residual_tol), encoding="utf-8"
    )
    (case_dir / "system" / "fvSchemes").write_text(_generate_fvschemes(), encoding="utf-8")
    (case_dir / "system" / "controlDict").write_text(
        _generate_control_dict(enable_wall_shear=True), encoding="utf-8"
    )

    _run_foam(case_dir, ["blockMesh"], "blockMesh.log")
    _run_foam(case_dir, ["simpleFoam"], "simpleFoam.log")
    _run_foam(case_dir, ["postProcess", "-func", "writeCellCentres", "-latestTime"], "postProcess_C.log")
    wall_shear_available = True
    try:
        _run_foam(case_dir, ["postProcess", "-func", "'wallShearStress(U)'", "-latestTime"], "postProcess_wallShearStress.log")
    except RuntimeError as exc:
        wall_shear_available = False
        logger.warning("wallShearStress postProcess failed, using near-wall fallback: %s", exc)

    latest = find_latest_time(case_dir)
    if latest is None:
        raise FileNotFoundError(f"No converged time directory found in {case_dir}")
    U = read_vector_field(latest / "U")
    C = read_cell_centres(case_dir)
    U_bar = _depth_average_velocity(C=C, U=U, x_round_decimals=9, y_round_decimals=9)
    tau_proxy = (6.0 * mu * np.linalg.norm(U_bar[:, 2:4], axis=1)) / H_m

    floor_wss_mag = np.array([])
    if wall_shear_available:
        wss_file = latest / "wallShearStress"
        floor_wss_mag = _parse_floor_wall_shear_magnitudes(wss_file)
    if floor_wss_mag.size == 0:
        floor_wss_mag = _estimate_floor_shear_from_near_wall_cells(C=C, U=U, mu=mu)
        wall_shear_available = False

    chamber_mask = (U_bar[:, 0] >= 0.0) & (U_bar[:, 0] <= L_m)
    developed_mask = chamber_mask & (U_bar[:, 0] >= 1.0e-3) & (U_bar[:, 0] <= (L_m - 1.0e-3))
    side_buffer = 2.0 * H_m
    core_mask = developed_mask & (U_bar[:, 1] >= side_buffer) & (U_bar[:, 1] <= (W_m - side_buffer))

    marker = case_dir / "case.foam"
    marker.write_text("", encoding="utf-8")

    # floor_wss_mag is boundary-face sampled; U_bar fields are cell-averaged.
    # We report robust summary stats from each without forcing a one-to-one map.
    results = {
        "case_dir": str(case_dir),
        "W_um": W_um,
        "H_um": H_um,
        "L_um": L_um,
        "Q_ul_min": Q_ul_min,
        "Re": float((rho * (Q_m3s / (W_m * H_m)) * H_m) / mu),
        "cv_global_3d_resolved": _safe_cv(floor_wss_mag),
        "cv_global_3d_proxy": _safe_cv(tau_proxy[chamber_mask]),
        "cv_developed_3d_proxy": _safe_cv(tau_proxy[developed_mask]),
        "cv_core_3d_proxy": _safe_cv(tau_proxy[core_mask]),
        "tau_floor_3d_mean": float(np.mean(floor_wss_mag)),
        "tau_floor_3d_min": float(np.min(floor_wss_mag)),
        "tau_floor_3d_max": float(np.max(floor_wss_mag)),
        "wall_shear_direct_available": wall_shear_available,
        "tau_floor_2d_proxy_mean": float(np.mean(tau_proxy[chamber_mask])),
        "depth_avg_samples": {
            "x_m": [float(v) for v in U_bar[:, 0]],
            "y_m": [float(v) for v in U_bar[:, 1]],
            "u_mag": [float(v) for v in np.linalg.norm(U_bar[:, 2:4], axis=1)],
        },
    }
    (case_dir / "metrics_3d.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def compare_2d_vs_3d_matched(
    case_2d_dir: Path,
    case_3d_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate head-to-head profile and scatter plots for matched 2D vs 3D."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_2d = find_latest_time(Path(case_2d_dir))
    latest_3d = find_latest_time(Path(case_3d_dir))
    if latest_2d is None or latest_3d is None:
        raise FileNotFoundError("Missing latest time for 2D or 3D case")

    U2 = read_vector_field(latest_2d / "U")
    C2 = read_cell_centres(Path(case_2d_dir))
    U3 = read_vector_field(latest_3d / "U")
    C3 = read_cell_centres(Path(case_3d_dir))
    U3bar = _depth_average_velocity(C=C3, U=U3, x_round_decimals=9, y_round_decimals=9)

    # Compare y-profiles at three x-stations from 2D against depth-averaged 3D.
    x2_unique = np.unique(np.round(C2[:, 0], 9))
    if x2_unique.size == 0:
        raise RuntimeError("No 2D x samples found")
    x_min, x_max = float(np.min(x2_unique)), float(np.max(x2_unique))
    x_targets = [x_min + 0.1 * (x_max - x_min), x_min + 0.5 * (x_max - x_min), x_min + 0.9 * (x_max - x_min)]
    profile_path = output_dir / "paper_ready_2d_vs_3d_yprofile.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, x_target in zip(axes, x_targets):
        x2 = x2_unique[np.argmin(np.abs(x2_unique - x_target))]
        x3 = np.unique(np.round(U3bar[:, 0], 9))
        x3_near = x3[np.argmin(np.abs(x3 - x_target))]

        mask2 = np.isclose(np.round(C2[:, 0], 9), x2)
        y2 = C2[mask2, 1]
        u2 = np.linalg.norm(U2[mask2, :2], axis=1)
        order2 = np.argsort(y2)

        mask3 = np.isclose(np.round(U3bar[:, 0], 9), x3_near)
        y3 = U3bar[mask3, 1]
        u3 = np.linalg.norm(U3bar[mask3, 2:4], axis=1)
        order3 = np.argsort(y3)

        ax.plot(y2[order2], u2[order2], label="2D U(x,y)")
        ax.plot(y3[order3], u3[order3], label="3D depth-avg U")
        ax.set_title(f"x ~ {x_target*1e3:.2f} mm")
        ax.set_xlabel("y (m)")
    axes[0].set_ylabel("|U| (m/s)")
    axes[0].legend()
    fig.savefig(profile_path, dpi=220)
    plt.close(fig)

    # Scatter + Bland-Altman on nearest-neighbor mapped U magnitudes.
    mapped_2d, mapped_3d = _nearest_map_2d_3d_u(C2=C2, U2=U2, U3bar=U3bar)
    scatter_path = output_dir / "paper_ready_2d_vs_3d_scatter_bland_altman.png"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].scatter(mapped_2d, mapped_3d, s=4, alpha=0.5)
    axes[0].set_xlabel("2D |U| (m/s)")
    axes[0].set_ylabel("3D depth-avg |U| (m/s)")
    axes[0].set_title("2D vs 3D (matched points)")
    mean_pair = 0.5 * (mapped_2d + mapped_3d)
    diff_pair = mapped_3d - mapped_2d
    axes[1].scatter(mean_pair, diff_pair, s=4, alpha=0.5)
    axes[1].axhline(np.mean(diff_pair), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Mean |U| (m/s)")
    axes[1].set_ylabel("3D - 2D (m/s)")
    axes[1].set_title("Bland-Altman")
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)

    return {
        "profile_plot": str(profile_path),
        "scatter_bland_altman_plot": str(scatter_path),
        "u_abs_mean_diff": float(np.mean(mapped_3d - mapped_2d)),
        "u_abs_rmse": float(np.sqrt(np.mean((mapped_3d - mapped_2d) ** 2))),
    }


def run_3d_validation(
    params: dict[str, float],
    pillar_config: str,
    H: float,
    config: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """Compatibility wrapper for existing module API."""
    if pillar_config != "none":
        raise ValueError("Minimal 3D sanity path only supports pillar_config='none'")
    W_um = float(params["W"])
    Q_ul_min = float(params["Q"])
    mu = float(config["fixed_parameters"]["fluid_viscosity_Pa_s"])
    rho = float(config["fixed_parameters"]["fluid_density_kg_m3"])
    L_um = float(config["fixed_parameters"]["chamber_length_um"])
    return run_3d_matched_rectangle(
        W_um=W_um,
        H_um=float(H),
        L_um=L_um,
        Q_ul_min=Q_ul_min,
        mu=mu,
        rho=rho,
        work_dir=output_dir,
    )


def compare_2d_vs_3d(
    metrics_2d: dict[str, float],
    metrics_3d: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Compatibility helper to compare scalar summary metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "cv_global_delta": float(metrics_3d.get("cv_global_3d_proxy", np.nan) - metrics_2d.get("cv_global", np.nan)),
        "cv_developed_delta": float(metrics_3d.get("cv_developed_3d_proxy", np.nan) - metrics_2d.get("cv_developed", np.nan)),
        "tau_mean_delta": float(metrics_3d.get("tau_floor_2d_proxy_mean", np.nan) - metrics_2d.get("tau_mean", np.nan)),
    }
    out = output_dir / "metrics_2d_vs_3d_summary.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def plot_3d_wss_contour(case_dir: Path, output_path: Path) -> Path:
    """Create histogram-style summary for resolved floor wall shear magnitudes."""
    case_dir = Path(case_dir)
    output_path = Path(output_path)
    latest = find_latest_time(case_dir)
    if latest is None:
        raise FileNotFoundError(f"No latest time directory in {case_dir}")
    magnitudes = _parse_floor_wall_shear_magnitudes(latest / "wallShearStress")
    if magnitudes.size == 0:
        raise RuntimeError("No floor wallShearStress values parsed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.hist(magnitudes, bins=40)
    plt.xlabel("|wallShearStress| (m^2/s^2)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def plot_streamlines(case_dir: Path, output_path: Path) -> Path:
    """Plot depth-averaged velocity quiver as a streamlines proxy."""
    case_dir = Path(case_dir)
    latest = find_latest_time(case_dir)
    if latest is None:
        raise FileNotFoundError(f"No latest time directory in {case_dir}")
    U = read_vector_field(latest / "U")
    C = read_cell_centres(case_dir)
    U_bar = _depth_average_velocity(C=C, U=U, x_round_decimals=9, y_round_decimals=9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.quiver(U_bar[:, 0], U_bar[:, 1], U_bar[:, 2], U_bar[:, 3], scale=None, width=0.002)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _prepare_3d_case(case_dir: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)


def _run_foam(case_dir: Path, cmd: list[str], log_name: str) -> None:
    prefix = _find_openfoam_prefix()
    full_cmd = [prefix, "-c", " ".join(cmd)] if prefix else cmd
    result = subprocess.run(
        full_cmd,
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ,
    )
    logs_dir = case_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / log_name).write_text(
        f"$ {' '.join(full_cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenFOAM command failed: {' '.join(full_cmd)}\n{result.stderr[-500:]}")


def _generate_3d_blockmesh_dict(L_m: float, W_m: float, H_m: float, nx: int, ny: int, nz: int) -> str:
    L_mm = L_m * 1e3
    W_mm = W_m * 1e3
    H_mm = H_m * 1e3
    return textwrap.dedent(
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
            (0 0 0)
            ({L_mm} 0 0)
            ({L_mm} {W_mm} 0)
            (0 {W_mm} 0)
            (0 0 {H_mm})
            ({L_mm} 0 {H_mm})
            ({L_mm} {W_mm} {H_mm})
            (0 {W_mm} {H_mm})
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
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
            floor
            {{
                type wall;
                faces ((0 3 2 1));
            }}
            ceiling
            {{
                type wall;
                faces ((4 5 6 7));
            }}
        );

        mergePatchPairs ();
        """
    )


def _generate_3d_u_file(u_inlet: float) -> str:
    return textwrap.dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       volVectorField;
            object      U;
        }}

        dimensions      [0 1 -1 0 0 0 0];
        internalField   uniform ({u_inlet} 0 0);

        boundaryField
        {{
            inlet
            {{
                type            fixedValue;
                value           uniform ({u_inlet} 0 0);
            }}
            outlet
            {{
                type            zeroGradient;
            }}
            walls
            {{
                type            noSlip;
            }}
            floor
            {{
                type            noSlip;
            }}
            ceiling
            {{
                type            noSlip;
            }}
        }}
        """
    )


def _generate_3d_p_file() -> str:
    return textwrap.dedent(
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
            floor
            {
                type            zeroGradient;
            }
            ceiling
            {
                type            zeroGradient;
            }
        }
        """
    )


def _generate_transport_properties(nu: float) -> str:
    return textwrap.dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      transportProperties;
        }}

        transportModel  Newtonian;
        nu              [0 2 -1 0 0 0 0] {nu};
        """
    )


def _generate_turbulence_properties() -> str:
    return textwrap.dedent(
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            object      turbulenceProperties;
        }

        simulationType  laminar;
        """
    )


def _generate_fvsolution(residual_tol: float) -> str:
    return textwrap.dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      fvSolution;
        }}

        solvers
        {{
            p
            {{
                solver          GAMG;
                smoother        GaussSeidel;
                tolerance       1e-8;
                relTol          0.01;
            }}
            U
            {{
                solver          smoothSolver;
                smoother        GaussSeidel;
                tolerance       1e-8;
                relTol          0.01;
            }}
        }}

        SIMPLE
        {{
            nNonOrthogonalCorrectors 1;
            consistent yes;
            residualControl
            {{
                U {residual_tol:.1e};
                p {residual_tol:.1e};
            }}
        }}

        relaxationFactors
        {{
            equations
            {{
                U 0.7;
                p 0.3;
            }}
        }}
        """
    )


def _generate_fvschemes() -> str:
    return textwrap.dedent(
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            object      fvSchemes;
        }

        ddtSchemes
        {
            default steadyState;
        }

        gradSchemes
        {
            default Gauss linear;
        }

        divSchemes
        {
            default none;
            div(phi,U) bounded Gauss linearUpwind grad(U);
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
        }

        laplacianSchemes
        {
            default Gauss linear corrected;
        }

        interpolationSchemes
        {
            default linear;
        }

        snGradSchemes
        {
            default corrected;
        }
        """
    )


def _generate_control_dict(enable_wall_shear: bool) -> str:
    functions = ""
    if enable_wall_shear:
        functions = textwrap.dedent(
            """\
            functions
            {
                wallShear
                {
                    type wallShearStress;
                    libs ("libfieldFunctionObjects.so");
                    patches (floor);
                    writeControl timeStep;
                    writeInterval 1;
                }
            }
            """
        )
    return textwrap.dedent(
        f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      controlDict;
        }}

        application simpleFoam;
        startFrom startTime;
        startTime 0;
        stopAt endTime;
        endTime 1500;
        deltaT 1;
        writeControl timeStep;
        writeInterval 1500;
        purgeWrite 1;
        writeFormat ascii;
        writePrecision 8;
        writeCompression off;
        timeFormat general;
        timePrecision 6;
        runTimeModifiable true;
        {functions}
        """
    )


def _depth_average_velocity(
    C: np.ndarray,
    U: np.ndarray,
    x_round_decimals: int,
    y_round_decimals: int,
) -> np.ndarray:
    # Returns Nx4 rows: [x, y, Ux_bar, Uy_bar]
    if C.shape[0] != U.shape[0]:
        raise ValueError("C/U size mismatch in depth-average computation")
    keys = np.column_stack([np.round(C[:, 0], x_round_decimals), np.round(C[:, 1], y_round_decimals)])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((unique_keys.shape[0], 4), dtype=float)
    for idx, (xk, yk) in enumerate(unique_keys):
        mask = inverse == idx
        out[idx, 0] = float(np.mean(C[mask, 0]))
        out[idx, 1] = float(np.mean(C[mask, 1]))
        out[idx, 2] = float(np.mean(U[mask, 0]))
        out[idx, 3] = float(np.mean(U[mask, 1]))
    return out


def _parse_floor_wall_shear_magnitudes(wss_path: Path) -> np.ndarray:
    if not wss_path.exists():
        return np.array([])
    text = wss_path.read_text(encoding="utf-8")
    patch_match = re.search(r"floor\s*\{(.*?)\n\s*\}", text, flags=re.DOTALL)
    if patch_match is None:
        return np.array([])
    patch_text = patch_match.group(1)
    nonuniform = re.search(
        r"value\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        patch_text,
        flags=re.DOTALL,
    )
    if nonuniform is None:
        return np.array([])
    vectors_raw = re.findall(
        r"\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)",
        nonuniform.group(1),
    )
    if not vectors_raw:
        return np.array([])
    vectors = np.array([[float(a), float(b), float(c)] for a, b, c in vectors_raw], dtype=float)
    return np.linalg.norm(vectors, axis=1)


def _nearest_map_2d_3d_u(C2: np.ndarray, U2: np.ndarray, U3bar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u2 = np.linalg.norm(U2[:, :2], axis=1)
    u3 = np.linalg.norm(U3bar[:, 2:4], axis=1)

    # Fast path: match on rounded (x, y) keys shared between 2D and 3D depth-avg grids.
    key2 = np.column_stack([np.round(C2[:, 0], 9), np.round(C2[:, 1], 9)])
    key3 = np.column_stack([np.round(U3bar[:, 0], 9), np.round(U3bar[:, 1], 9)])
    map3 = {(float(x), float(y)): i for i, (x, y) in enumerate(key3)}
    idx2 = []
    idx3 = []
    for i, (xk, yk) in enumerate(key2):
        j = map3.get((float(xk), float(yk)))
        if j is not None:
            idx2.append(i)
            idx3.append(j)
    if idx2:
        return u2[np.asarray(idx2, dtype=int)], u3[np.asarray(idx3, dtype=int)]

    # Fallback for mismatched grids: sample at most 2000 points from 2D and do chunked NN.
    n = C2.shape[0]
    if n > 2000:
        sample_idx = np.linspace(0, n - 1, 2000).astype(int)
    else:
        sample_idx = np.arange(n, dtype=int)
    q = C2[sample_idx, :2]
    r = U3bar[:, :2]
    mapped = []
    chunk = 250
    for i in range(0, q.shape[0], chunk):
        qq = q[i : i + chunk]
        # squared distances: (m, k)
        d2 = np.sum((qq[:, None, :] - r[None, :, :]) ** 2, axis=2)
        mapped.append(np.argmin(d2, axis=1))
    nn = np.concatenate(mapped)
    return u2[sample_idx], u3[nn]


def _estimate_floor_shear_from_near_wall_cells(C: np.ndarray, U: np.ndarray, mu: float) -> np.ndarray:
    # Fallback when wallShearStress function object is unavailable: estimate
    # tau_w ~ mu * |u_t| / z for the first cell center above the floor.
    keys = np.column_stack([np.round(C[:, 0], 9), np.round(C[:, 1], 9)])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    tau_vals = []
    for idx in range(unique_keys.shape[0]):
        mask = inverse == idx
        z_vals = C[mask, 2]
        if z_vals.size == 0:
            continue
        j_local = int(np.argmin(z_vals))
        u_t = np.linalg.norm(U[mask][j_local, :2])
        z = float(max(z_vals[j_local], 1e-12))
        tau_vals.append(mu * (u_t / z))
    return np.asarray(tau_vals, dtype=float)


def _safe_cv(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mean = float(np.mean(values))
    if abs(mean) < 1e-30:
        return float("nan")
    return float(np.std(values) / mean)
