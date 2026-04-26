"""
Phase 1 — single-shot CFD on a NEW topology candidate: an N=8 y-stacked ladder
of inlet patches at x=0, each prescribed C_k = k/(N-1).  Tests whether the
chamber can preserve a transverse linear gradient (axis=y target) under flow.

This is a STANDALONE diagnostic: it does not modify the production CFD pipeline
or BO infrastructure.  It:

    1. Generates a custom blockMeshDict with N inlet patches at x=0.
    2. Copies the template OpenFOAM case and overwrites 0/U, 0/p, 0/T with
       N-inlet boundary conditions (uniform U_x per inlet, C_k per inlet).
    3. Runs blockMesh + simpleFoam + scalarTransportFoam.
    4. Reads the C field and computes L2_to_target against `linear_gradient
       axis=y, c_high=1, c_low=0`.

If L2 < ~0.3 here, the topology can in principle achieve a y-gradient and
Phase 2 (a fresh BO campaign on this topology) is justified.  If L2 > ~0.5,
the chamber's diffusion length is too long and the gradient washes out
before the outlet — the topology needs geometric changes (shorter chamber,
faster flow, lower diffusivity) before BO is worthwhile.

Output:
    examples/tumor_chip_linear_gradient/data/diagnostic/ladder/baseline_metrics.json
    examples/tumor_chip_linear_gradient/data/diagnostic/ladder/case/  (full OpenFOAM case)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ooc_optimizer.geometry.topology_blockmesh import _render
from ooc_optimizer.cfd.scalar import (
    _find_openfoam_prefix,
    extract_concentration_field,
    run_scalar_transport,
)
from ooc_optimizer.optimization.objectives import build_target_profile, l2_to_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("diagnostic_ladder")


def build_ladder_blockmesh(N: int, W_mm: float, L_mm: float, dz_mm: float,
                           base_nx: int, ny_per_mm: int) -> tuple[str, list[str], list[float]]:
    """Construct a ladder blockMeshDict and return (text, inlet_names, inlet_y_centres_mm)."""
    if N < 2:
        raise ValueError(f"Ladder requires N >= 2 (got {N})")
    if W_mm <= 0 or L_mm <= 0 or dz_mm <= 0:
        raise ValueError("W_mm, L_mm, dz_mm must all be positive")

    ys = [k * W_mm / N for k in range(N + 1)]
    xs = [0.0, L_mm]

    vertices: list[tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * dz_mm
        for iy in range(N + 1):
            for ix in (0, 1):
                vertices.append((xs[ix], ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * (N + 1) * 2 + iy * 2 + ix

    blocks = []
    for iy in range(N):
        strip_w = ys[iy + 1] - ys[iy]
        ny = max(2, int(round(strip_w * ny_per_mm)))
        verts = (
            v(0, iy, 0), v(1, iy, 0), v(1, iy + 1, 0), v(0, iy + 1, 0),
            v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1),
        )
        blocks.append((verts, (base_nx, ny, 1)))

    x0_face = lambda iy: (v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0))
    x1_face = lambda iy: (v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1))
    ymin_face = (v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1))
    ymax_face = (v(0, N, 0), v(0, N, 1), v(1, N, 1), v(1, N, 0))
    front_face = lambda iy: (v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0))
    back_face = lambda iy: (v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1))

    inlet_names = [f"inlet_{k}" for k in range(N)]
    patches: dict = {name: ("patch", [x0_face(k)]) for k, name in enumerate(inlet_names)}
    patches["outlet"] = ("patch", [x1_face(iy) for iy in range(N)])
    patches["walls"] = ("wall", [])
    patches["floor"] = ("wall", [ymin_face, ymax_face])
    patches["frontAndBack"] = (
        "empty",
        [front_face(iy) for iy in range(N)] + [back_face(iy) for iy in range(N)],
    )

    text = _render(vertices, blocks, patches)
    y_centres_mm = [0.5 * (ys[k] + ys[k + 1]) for k in range(N)]
    return text, inlet_names, y_centres_mm


def write_u_field(case_dir: Path, inlet_names: list[str], U_in: float) -> None:
    bc_blocks = []
    for name in inlet_names:
        bc_blocks.append(textwrap.dedent(f"""\
            {name}
            {{
                type            fixedValue;
                value           uniform ({U_in:.6f} 0 0);
            }}
        """))
    bc_blocks.extend([
        "outlet\n{ type zeroGradient; }\n",
        "walls\n{ type noSlip; }\n",
        "floor\n{ type noSlip; }\n",
        "frontAndBack\n{ type empty; }\n",
    ])
    content = textwrap.dedent("""\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volVectorField;
            object      U;
        }

        dimensions      [0 1 -1 0 0 0 0];

        internalField   uniform (0 0 0);

        boundaryField
        {
        """)
    content += "    " + "\n    ".join("\n".join(bc_blocks).splitlines()) + "\n}\n"
    (case_dir / "0" / "U").write_text(content)


def write_p_field(case_dir: Path, inlet_names: list[str]) -> None:
    bc_blocks = [f"{n}\n{{ type zeroGradient; }}\n" for n in inlet_names]
    bc_blocks.extend([
        "outlet\n{ type fixedValue; value uniform 0; }\n",
        "walls\n{ type zeroGradient; }\n",
        "floor\n{ type zeroGradient; }\n",
        "frontAndBack\n{ type empty; }\n",
    ])
    content = textwrap.dedent("""\
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
        """)
    content += "    " + "\n    ".join("\n".join(bc_blocks).splitlines()) + "\n}\n"
    (case_dir / "0" / "p").write_text(content)


def write_t_field(case_dir: Path, inlet_names: list[str], C_values: list[float]) -> None:
    if len(inlet_names) != len(C_values):
        raise ValueError("inlet_names and C_values length mismatch")
    bc_blocks = [
        f"{n}\n{{ type fixedValue; value uniform {c:.6f}; }}\n"
        for n, c in zip(inlet_names, C_values)
    ]
    bc_blocks.extend([
        "outlet\n{ type zeroGradient; }\n",
        "walls\n{ type zeroGradient; }\n",
        "floor\n{ type zeroGradient; }\n",
        "frontAndBack\n{ type empty; }\n",
    ])
    content = textwrap.dedent("""\
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
        """)
    content += "    " + "\n    ".join("\n".join(bc_blocks).splitlines()) + "\n}\n"
    (case_dir / "0" / "T").write_text(content)


def run_blockmesh(case_dir: Path) -> None:
    prefix = _find_openfoam_prefix()
    cmd = ["blockMesh"] if not prefix else [prefix, "-c", "blockMesh"]
    log_path = case_dir / "logs" / "blockMesh.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        subprocess.run(cmd, cwd=case_dir, stdout=f, stderr=subprocess.STDOUT, check=True, env=os.environ)


def run_simplefoam(case_dir: Path, timeout: int = 600) -> None:
    prefix = _find_openfoam_prefix()
    cmd = ["simpleFoam"] if not prefix else [prefix, "-c", "simpleFoam"]
    log_path = case_dir / "simpleFoam.log"
    with open(log_path, "w") as f:
        subprocess.run(cmd, cwd=case_dir, stdout=f, stderr=subprocess.STDOUT, check=True, env=os.environ, timeout=timeout)


def run_postprocess(case_dir: Path, func: str) -> None:
    prefix = _find_openfoam_prefix()
    cmd = ["postProcess", "-func", func, "-latestTime"]
    if prefix:
        cmd = [prefix, "-c", " ".join(cmd)]
    subprocess.run(cmd, cwd=case_dir, capture_output=True, check=True, env=os.environ)


def main() -> int:
    # Geometry & flow
    N = int(os.environ.get("LADDER_N", "8"))
    W_mm = 3.0          # 3 mm chamber width (matches default mid-bound)
    L_mm = 10.0         # 10 mm chamber length (fixed in default config)
    H_um = 200.0        # 200 μm chamber height
    dz_mm = H_um / 1000.0  # blockMesh in mm; full 3D height as one z-cell
    Q_total_uL_min = 50.0
    diffusivity_m2s = 1.0e-10
    base_nx = 200
    ny_per_mm = 25

    # Build ladder mesh and BCs
    bm_text, inlet_names, y_centres_mm = build_ladder_blockmesh(
        N=N, W_mm=W_mm, L_mm=L_mm, dz_mm=dz_mm, base_nx=base_nx, ny_per_mm=ny_per_mm,
    )
    # CONVENTION:
    #   "endpoint"  C_k = k/(N-1)   — strip 0 = 0, strip N-1 = 1.  Inlets land
    #               on the *boundaries* of the target ramp, leaving a ±1/(2N)
    #               offset at the strip CENTRES (where the residual L2 is
    #               actually measured).
    #   "midpoint"  C_k = (k+0.5)/N — inlets land on the strip centres, which
    #               match the linear target exactly.  Within-strip variation
    #               (target ramps but C_k is constant) is the only residual.
    convention = os.environ.get("LADDER_CONVENTION", "endpoint")
    if convention == "midpoint":
        C_values = [(k + 0.5) / N for k in range(N)]
    elif convention == "endpoint":
        C_values = [k / (N - 1) for k in range(N)]
    else:
        raise ValueError(f"Unknown LADDER_CONVENTION '{convention}'")
    log.info("Ladder N=%d convention=%s, C_k=%s", N, convention, [f"{c:.3f}" for c in C_values])

    # Inlet velocity: each inlet has equal area (W/N) * dz, and total flow Q is
    # split equally → U_in is the same uniform x-velocity for every inlet.
    Q_total_m3s = Q_total_uL_min * 1e-9 / 60.0
    inlet_area_m2 = (W_mm / N * 1e-3) * (dz_mm * 1e-3)
    U_in = (Q_total_m3s / N) / inlet_area_m2
    log.info("Per-inlet U_x = %.6f m/s (Q_total=%.2f μL/min split across %d inlets)", U_in, Q_total_uL_min, N)

    # Stage case
    out_root = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "diagnostic" / "ladder"
    case_dir = out_root / "case"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    template_dir = REPO_ROOT / "ooc_optimizer" / "cfd" / "template"
    shutil.copytree(template_dir, case_dir)
    (case_dir / "system" / "blockMeshDict").write_text(bm_text)
    write_u_field(case_dir, inlet_names, U_in)
    write_p_field(case_dir, inlet_names)
    write_t_field(case_dir, inlet_names, C_values)

    # Mesh + momentum
    log.info("blockMesh…")
    run_blockmesh(case_dir)
    log.info("simpleFoam…")
    run_simplefoam(case_dir, timeout=600)
    log.info("writeCellCentres…")
    run_postprocess(case_dir, "writeCellCentres")

    # Scalar transport — reuse production helper, override patch BCs for our N inlets
    log.info("scalarTransportFoam…")
    scalar_patches = {
        **{name: f"fixedValue:{c}" for name, c in zip(inlet_names, C_values)},
        "outlet": "zeroGradient",
        "walls": "zeroGradient",
        "floor": "zeroGradient",
        "frontAndBack": "empty",
    }
    scalar_result = run_scalar_transport(
        case_dir=case_dir,
        diffusivity_m2_s=diffusivity_m2s,
        c_drug=1.0, c_medium=0.0,        # ignored when patches= is given
        patches=scalar_patches,
    )
    log.info("scalar converged=%s warnings=%s", scalar_result.converged, scalar_result.warnings)

    # Read field, compute L2 against axis=y linear gradient
    centres, C = extract_concentration_field(case_dir)
    L_m = L_mm * 1e-3
    W_m = W_mm * 1e-3
    target = build_target_profile({"kind": "linear_gradient", "axis": "y", "c_high": 1.0, "c_low": 0.0})
    C_target = target.evaluate(centres[:, 0], centres[:, 1], L=L_m, W=W_m)
    L2 = float(l2_to_target(C, C_target))

    # Stats for context
    stats = {
        "L2_to_target_axis_y": L2,
        "C_mean": float(np.mean(C)),
        "C_std": float(np.std(C)),
        "C_min": float(np.min(C)),
        "C_max": float(np.max(C)),
        "n_cells": int(len(C)),
        "L_mm": L_mm, "W_mm": W_mm, "H_um": H_um, "N_inlets": N,
        "Q_total_uL_min": Q_total_uL_min,
        "U_in_m_s": U_in,
        "diffusivity_m2_s": diffusivity_m2s,
        "scalar_converged": bool(scalar_result.converged),
        "case_dir": str(case_dir),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "baseline_metrics.json").write_text(json.dumps(stats, indent=2))

    print()
    print("=== Phase 1: ladder N=8 baseline vs linear_gradient axis=y ===")
    for k, v in stats.items():
        print(f"  {k:<26} = {v}")
    print()
    if L2 < 0.3:
        print(f"VERDICT: L2={L2:.4f} < 0.3 → topology can produce a y-gradient. Proceed to Phase 2 BO.")
    elif L2 < 0.5:
        print(f"VERDICT: L2={L2:.4f} in [0.3, 0.5) → marginal; tune chamber length / Q before Phase 2.")
    else:
        print(f"VERDICT: L2={L2:.4f} ≥ 0.5 → gradient washes out; geometry needs rework before BO is worth it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
