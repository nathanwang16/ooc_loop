"""
Runner utilities for the WSS uniformity premise test.

This module compares matched and mismatched inlet configurations using 2D
simpleFoam runs and returns detailed spatial metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ooc_optimizer.cfd.foam_parser import find_latest_time, read_cell_centres, read_vector_field
from ooc_optimizer.cfd.stepped_blockmesh import generate_stepped_blockmesh_dict
from ooc_optimizer.cfd.verification import _find_openfoam_prefix

logger = logging.getLogger(__name__)

DEFAULT_W_IN_UM = 500.0
DEFAULT_L_CHAMBER_UM = 10000.0
DEFAULT_L_STUB_UM = 2000.0
DEFAULT_DZ_MM = 0.01


def run_premise_case(
    W_um: float,
    H_um: float,
    Q_ul_min: float,
    inlet_mode: str,
    work_dir: Path,
    mu: float,
    rho: float,
    residual_tol: float = 1e-6,
    l_chamber_um: float = DEFAULT_L_CHAMBER_UM,
    l_stub_um: float = DEFAULT_L_STUB_UM,
    w_in_um: float = DEFAULT_W_IN_UM,
    dz_mm: float = DEFAULT_DZ_MM,
    template_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one premise-test case and return metrics and profile data."""
    if inlet_mode not in {"matched", "mismatched"}:
        raise ValueError("inlet_mode must be one of {'matched', 'mismatched'}")
    if template_dir is None:
        template_dir = Path("ooc_optimizer/cfd/template")

    W_m = W_um * 1e-6
    H_m = H_um * 1e-6
    L_chamber_m = l_chamber_um * 1e-6
    L_stub_m = l_stub_um * 1e-6
    W_in_um_case = W_um if inlet_mode == "matched" else w_in_um
    W_in_m = W_in_um_case * 1e-6

    case_name = f"{inlet_mode}_W{int(W_um)}_H{int(H_um)}_Q{int(Q_ul_min)}"
    case_dir = Path(work_dir) / case_name
    _prepare_case_directory(case_dir=case_dir, template_dir=Path(template_dir))

    nx_chamber = 200
    ny_chamber = max(40, int(round(20.0 * (W_um / 1000.0))))
    nx_stub = 40
    ny_stub_in = max(8, int(round(20.0 * (W_in_um_case / max(W_um, 1e-9)))))

    bmd = generate_stepped_blockmesh_dict(
        L_chamber_mm=l_chamber_um / 1000.0,
        W_chamber_mm=W_um / 1000.0,
        W_in_mm=W_in_um_case / 1000.0,
        L_stub_mm=0.0 if inlet_mode == "matched" else (l_stub_um / 1000.0),
        dz_mm=dz_mm,
        nx_chamber=nx_chamber,
        ny_chamber=ny_chamber,
        nx_stub=nx_stub,
        ny_stub_in=ny_stub_in,
    )
    (case_dir / "system" / "blockMeshDict").write_text(bmd, encoding="utf-8")
    _set_residual_controls(case_dir=case_dir, residual_tol=residual_tol)

    Q_m3s = Q_ul_min * 1e-9 / 60.0
    u_inlet = Q_m3s / (W_in_m * H_m)
    _set_inlet_velocity(case_dir=case_dir, u_inlet=u_inlet)

    started = time.time()
    _run_openfoam_tool(case_dir=case_dir, cmd=["blockMesh"], log_name="blockMesh.log")
    _run_openfoam_tool(case_dir=case_dir, cmd=["simpleFoam"], log_name="simpleFoam.log")
    _run_openfoam_tool(
        case_dir=case_dir,
        cmd=["postProcess", "-func", "writeCellCentres", "-latestTime"],
        log_name="postProcess_writeCellCentres.log",
    )
    elapsed_s = time.time() - started

    converged, solver_iterations = _parse_simplefoam_convergence(
        case_dir / "logs" / "simpleFoam.log",
        threshold=residual_tol,
    )
    if not converged:
        raise RuntimeError(
            f"Case {case_name} did not meet residual threshold {residual_tol:.1e}"
        )

    metrics = _compute_premise_metrics(
        case_dir=case_dir,
        inlet_mode=inlet_mode,
        W_m=W_m,
        H_m=H_m,
        L_chamber_m=L_chamber_m,
        L_stub_m=L_stub_m,
        mu=mu,
        rho=rho,
        Q_m3s=Q_m3s,
    )
    metrics["solver_iterations"] = solver_iterations
    metrics["wall_clock_s"] = float(elapsed_s)
    metrics["case_dir"] = str(case_dir)
    metrics["mesh_n_cells"] = int(metrics.get("n_cells", 0))
    metrics["converged"] = True

    marker = case_dir / "case.foam"
    marker.write_text("", encoding="utf-8")

    metrics_path = case_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_grid(
    grid: dict[str, list[float]],
    inlet_mode: str,
    work_dir: Path,
    mu: float,
    rho: float,
    residual_tol: float = 1e-6,
    template_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a full grid for one inlet mode."""
    results: list[dict[str, Any]] = []
    for W_um in grid["W"]:
        for H_um in grid["H"]:
            for Q_ul_min in grid["Q"]:
                logger.info(
                    "Running premise case mode=%s W=%s H=%s Q=%s",
                    inlet_mode,
                    W_um,
                    H_um,
                    Q_ul_min,
                )
                result = run_premise_case(
                    W_um=W_um,
                    H_um=H_um,
                    Q_ul_min=Q_ul_min,
                    inlet_mode=inlet_mode,
                    work_dir=work_dir,
                    mu=mu,
                    rho=rho,
                    residual_tol=residual_tol,
                    template_dir=template_dir,
                )
                results.append(result)
    return results


def _prepare_case_directory(case_dir: Path, template_dir: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    shutil.copytree(template_dir, case_dir)


def _set_inlet_velocity(case_dir: Path, u_inlet: float) -> None:
    u_file = case_dir / "0" / "U"
    if not u_file.exists():
        raise FileNotFoundError(f"Missing U boundary file: {u_file}")
    content = u_file.read_text(encoding="utf-8")
    updated, n = re.subn(
        r"(inlet\s*\{.*?value\s+uniform\s+\()([^)]+)(\);)",
        rf"\g<1>{u_inlet:.12g} 0 0\g<3>",
        content,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise ValueError("Failed to inject inlet velocity into 0/U")
    u_file.write_text(updated, encoding="utf-8")


def _set_residual_controls(case_dir: Path, residual_tol: float) -> None:
    fv_solution = case_dir / "system" / "fvSolution"
    if not fv_solution.exists():
        raise FileNotFoundError(f"Missing fvSolution file: {fv_solution}")
    content = fv_solution.read_text(encoding="utf-8")
    updated = re.sub(
        r"(U\s+)([0-9eE\+\-\.]+)(\s*;)",
        rf"\g<1>{residual_tol:.1e}\g<3>",
        content,
        count=1,
    )
    updated = re.sub(
        r"(p\s+)([0-9eE\+\-\.]+)(\s*;)",
        rf"\g<1>{residual_tol:.1e}\g<3>",
        updated,
        count=1,
    )
    fv_solution.write_text(updated, encoding="utf-8")


def _run_openfoam_tool(case_dir: Path, cmd: list[str], log_name: str) -> subprocess.CompletedProcess:
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
        raise RuntimeError(f"OpenFOAM command failed ({' '.join(full_cmd)}): {result.stderr[-400:]}")
    return result


def _parse_simplefoam_convergence(log_path: Path, threshold: float) -> tuple[bool, int]:
    if not log_path.exists():
        raise FileNotFoundError(f"simpleFoam log missing: {log_path}")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"final residual\s*=\s*([0-9eE\+\-\.]+)", flags=re.IGNORECASE)
    residual_u = None
    residual_p = None
    iterations = 0
    for line in lines:
        if line.startswith("Time = "):
            iterations += 1
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        value = float(match.group(1))
        if ("Solving for Ux" in line or "Solving for Uy" in line) and residual_u is None:
            residual_u = value
        if "Solving for p" in line and residual_p is None:
            residual_p = value
        if residual_u is not None and residual_p is not None:
            break
    converged = (
        residual_u is not None
        and residual_p is not None
        and residual_u < threshold
        and residual_p < threshold
    )
    return converged, iterations


def _compute_premise_metrics(
    case_dir: Path,
    inlet_mode: str,
    W_m: float,
    H_m: float,
    L_chamber_m: float,
    L_stub_m: float,
    mu: float,
    rho: float,
    Q_m3s: float,
) -> dict[str, Any]:
    latest_time = find_latest_time(case_dir)
    if latest_time is None:
        raise FileNotFoundError(f"No result time directory found in {case_dir}")
    U = read_vector_field(latest_time / "U")
    C = read_cell_centres(case_dir)
    if U.shape[0] != C.shape[0]:
        raise ValueError("Mismatch between U and C cell counts")

    U_mag = np.linalg.norm(U[:, :2], axis=1)
    tau = (6.0 * mu * U_mag) / H_m
    x = C[:, 0]
    y = C[:, 1]

    chamber_x0 = 0.0 if inlet_mode == "matched" else L_stub_m
    chamber_x1 = chamber_x0 + L_chamber_m
    chamber_mask = (x >= chamber_x0) & (x <= chamber_x1)
    developed_mask = chamber_mask & (x >= (chamber_x0 + 1.0e-3)) & (x <= (chamber_x1 - 1.0e-3))
    sidewall_buffer = 2.0 * H_m
    core_mask = developed_mask & (y >= sidewall_buffer) & (y <= (W_m - sidewall_buffer))
    chamber_mid = chamber_x0 + 0.5 * L_chamber_m
    core_central_mask = core_mask & (x >= (chamber_mid - 3.0e-3)) & (x <= (chamber_mid + 3.0e-3))

    tau_mean = _safe_mean(tau[chamber_mask])
    tau_std = _safe_std(tau[chamber_mask])

    centerline_profile = _centerline_tau_profile(x=x, y=y, tau=tau, chamber_mask=chamber_mask, w_m=W_m)
    section_profiles = _section_tau_profiles(
        x=x,
        y=y,
        tau=tau,
        chamber_x0=chamber_x0,
        l_chamber_m=L_chamber_m,
        y_min=0.0,
        y_max=W_m,
    )
    cv_y_of_x = _cv_y_of_x(
        x=x,
        y=y,
        tau=tau,
        chamber_mask=chamber_mask,
        y_min=sidewall_buffer,
        y_max=W_m - sidewall_buffer,
    )
    cv_x_of_y = _cv_x_of_y(
        x=x,
        y=y,
        tau=tau,
        developed_mask=developed_mask,
    )

    entrance_length = _estimate_entrance_length(
        centerline_profile=centerline_profile,
        chamber_x0=chamber_x0,
        tau_core_mean=_safe_mean(tau[core_mask]),
    )

    reynolds = (rho * (Q_m3s / (W_m * H_m)) * H_m) / mu
    result = {
        "inlet_mode": inlet_mode,
        "W_um": W_m * 1e6,
        "H_um": H_m * 1e6,
        "Q_ul_min": Q_m3s * 60.0 * 1e9,
        "Re": float(reynolds),
        "n_cells": int(len(tau)),
        "cv_global": _safe_cv(tau[chamber_mask]),
        "cv_developed": _safe_cv(tau[developed_mask]),
        "cv_core": _safe_cv(tau[core_mask]),
        "cv_core_central": _safe_cv(tau[core_central_mask]),
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "tau_min": float(np.min(tau[chamber_mask])) if np.any(chamber_mask) else float("nan"),
        "tau_max": float(np.max(tau[chamber_mask])) if np.any(chamber_mask) else float("nan"),
        "f_dead": _compute_dead_fraction(U_mag[chamber_mask]),
        "centerline_tau_of_x": centerline_profile,
        "tau_profiles_by_x": section_profiles,
        "cv_y_of_x": cv_y_of_x,
        "cv_x_of_y": cv_x_of_y,
        "entrance_length_estimate_m": float(entrance_length),
        "chamber_x_bounds_m": [float(chamber_x0), float(chamber_x1)],
    }
    return result


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _safe_cv(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mean = float(np.mean(values))
    if abs(mean) < 1e-30:
        return float("nan")
    return float(np.std(values) / mean)


def _compute_dead_fraction(values: np.ndarray, threshold_ratio: float = 0.1) -> float:
    if values.size == 0:
        return float("nan")
    u_mean = float(np.mean(values))
    if abs(u_mean) < 1e-30:
        return 1.0
    return float(np.mean(values < (threshold_ratio * u_mean)))


def _centerline_tau_profile(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    chamber_mask: np.ndarray,
    w_m: float,
) -> dict[str, list[float]]:
    if not np.any(chamber_mask):
        return {"x_m": [], "tau_pa": []}
    x_ch = x[chamber_mask]
    y_ch = y[chamber_mask]
    tau_ch = tau[chamber_mask]
    x_bins = np.unique(np.round(x_ch, 9))
    y_center = 0.5 * w_m
    xs: list[float] = []
    ts: list[float] = []
    for xv in x_bins:
        mask_x = np.isclose(np.round(x_ch, 9), xv)
        if not np.any(mask_x):
            continue
        y_slice = y_ch[mask_x]
        tau_slice = tau_ch[mask_x]
        idx = int(np.argmin(np.abs(y_slice - y_center)))
        xs.append(float(np.mean(x_ch[mask_x])))
        ts.append(float(tau_slice[idx]))
    return {"x_m": xs, "tau_pa": ts}


def _section_tau_profiles(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    chamber_x0: float,
    l_chamber_m: float,
    y_min: float,
    y_max: float,
) -> dict[str, dict[str, list[float]]]:
    fractions = [0.1, 0.5, 0.9]
    x_unique = np.unique(np.round(x, 9))
    sections: dict[str, dict[str, list[float]]] = {}
    for frac in fractions:
        x_target = chamber_x0 + frac * l_chamber_m
        x_nearest = x_unique[np.argmin(np.abs(x_unique - x_target))]
        mask = np.isclose(np.round(x, 9), x_nearest) & (y >= y_min) & (y <= y_max)
        ys = y[mask]
        taus = tau[mask]
        order = np.argsort(ys)
        sections[f"x_{int(frac*100)}pct"] = {
            "y_m": [float(v) for v in ys[order]],
            "tau_pa": [float(v) for v in taus[order]],
            "cv_y": _safe_cv(taus),
        }
    return sections


def _cv_y_of_x(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    chamber_mask: np.ndarray,
    y_min: float,
    y_max: float,
) -> dict[str, list[float]]:
    local_mask = chamber_mask & (y >= y_min) & (y <= y_max)
    x_unique = np.unique(np.round(x[local_mask], 9))
    x_list: list[float] = []
    cv_list: list[float] = []
    for xv in x_unique:
        mask = local_mask & np.isclose(np.round(x, 9), xv)
        x_list.append(float(np.mean(x[mask])))
        cv_list.append(_safe_cv(tau[mask]))
    return {"x_m": x_list, "cv": cv_list}


def _cv_x_of_y(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    developed_mask: np.ndarray,
) -> dict[str, list[float]]:
    y_unique = np.unique(np.round(y[developed_mask], 9))
    y_list: list[float] = []
    cv_list: list[float] = []
    for yv in y_unique:
        mask = developed_mask & np.isclose(np.round(y, 9), yv)
        y_list.append(float(np.mean(y[mask])))
        cv_list.append(_safe_cv(tau[mask]))
    return {"y_m": y_list, "cv": cv_list}


def _estimate_entrance_length(centerline_profile: dict[str, list[float]], chamber_x0: float, tau_core_mean: float) -> float:
    xs = centerline_profile.get("x_m", [])
    ts = centerline_profile.get("tau_pa", [])
    if not xs or not ts or np.isnan(tau_core_mean) or abs(tau_core_mean) < 1e-30:
        return float("nan")
    for xv, tv in zip(xs, ts):
        if abs((tv - tau_core_mean) / tau_core_mean) < 0.05:
            return float(max(0.0, xv - chamber_x0))
    return float("nan")
