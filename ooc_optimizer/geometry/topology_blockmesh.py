"""
Module 1.2 / 2.1 — Topology-aware blockMeshDict generation.

For the 2D BO loop we mesh directly from blockMesh rather than via STL, so the
inlet-patch naming contract required by the v2 solver pipeline is enforced
here in a single place.  Each topology produces a valid blockMeshDict with
patches named exactly:

    inlet_drug, inlet_medium, outlet, walls, floor, frontAndBack

Topologies:
    opposing        — two separated short-side inlets at x = 0 with a PDMS
                      tongue between them (y-stacked multi-block mesh).
    same_side_Y     — single upstream Y-junction resolved as two half-height
                      inlet patches at x = 0 (y-stacked multi-block mesh).
    asymmetric_lumen— medium enters on x = 0, drug enters on a 60% strip of
                      the y = 0 wall (mixed short-edge / long-edge inlets).

Coordinates: blockMesh is written in mm (convertToMeters 0.001), x ∈ [0, L],
y ∈ [0, W], z ∈ [0, dz].  The floor is y = 0; walls include y = W and any
non-inlet segment of y = 0.  frontAndBack is empty for the one-cell-thick
slab used in 2D CFD.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

VALID_TOPOLOGIES = {"opposing", "same_side_Y", "asymmetric_lumen"}

L_CHAMBER_MM = 10.0
W_INLET_MM = 0.5  # 500 μm standard inlet width
DZ_MM_DEFAULT = 0.01


@dataclass
class BlockMeshResult:
    """Everything the downstream solver needs to stage a case."""

    content: str
    inlet_drug_area_m2: float
    inlet_medium_area_m2: float
    chamber_length_m: float
    chamber_width_m: float
    chamber_height_m: float
    base_nx: int
    base_ny: int


def generate_blockmesh_dict_v2(
    *,
    params: Dict[str, float],
    topology: str,
    H_um: float,
    dz_mm: float = DZ_MM_DEFAULT,
    base_nx: int = 200,
    ny_per_mm: int = 20,
) -> BlockMeshResult:
    """Dispatch to the topology-specific blockMeshDict builder (2D slab).

    Returns the textual blockMeshDict plus the inlet areas (used by
    ``ooc_optimizer.cfd.solver`` to compute per-inlet velocities from
    ``Q_total`` and ``r_flow``).
    """
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topology}'")
    W_mm = float(params["W"]) / 1000.0
    if topology == "opposing":
        delta_W = float(params["delta_W"])
        return _bm_opposing(
            W_mm=W_mm,
            dz_mm=dz_mm,
            base_nx=base_nx,
            ny_per_mm=ny_per_mm,
            delta_W=delta_W,
        )
    if topology == "same_side_Y":
        return _bm_same_side_y(W_mm=W_mm, dz_mm=dz_mm, base_nx=base_nx, ny_per_mm=ny_per_mm)
    return _bm_asymmetric_lumen(
        W_mm=W_mm,
        dz_mm=dz_mm,
        base_nx=base_nx,
        ny_per_mm=ny_per_mm,
    )


def generate_blockmesh_dict_v2_3d(
    *,
    params: Dict[str, float],
    topology: str,
    H_um: float,
    base_nx: int = 300,
    ny_per_mm: int = 30,
    nz: int = 25,
    z_grading: float = 1.0,
) -> BlockMeshResult:
    """3D counterpart of :func:`generate_blockmesh_dict_v2`.

    Wraps the same xy multi-block layout but extrudes in z from 0 to ``H_um``
    with ``nz`` layers.  The z = 0 face is tagged ``floor`` (culture surface),
    z = H is tagged ``ceiling``; frontAndBack becomes physical no-slip walls
    instead of ``empty``.  ``z_grading`` is the OpenFOAM end/start ratio for
    the z-edges (use < 1 to concentrate cells near the floor; 1 for uniform).

    This is used by Module 4.1 (3D CFD validation) to run simpleFoam and
    scalarTransportFoam on the BO-winning geometry at higher fidelity than
    the 2D slab used during BO.
    """
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topology}'")
    if H_um <= 0:
        raise ValueError("H_um must be positive")
    if nz < 4:
        raise ValueError("nz must be >= 4 for 3D validation")
    W_mm = float(params["W"]) / 1000.0
    H_mm = H_um / 1000.0

    if topology == "opposing":
        res = _bm_opposing_3d(
            W_mm=W_mm,
            H_mm=H_mm,
            base_nx=base_nx,
            ny_per_mm=ny_per_mm,
            nz=nz,
            z_grading=z_grading,
            delta_W=float(params["delta_W"]),
        )
    elif topology == "same_side_Y":
        res = _bm_same_side_y_3d(
            W_mm=W_mm,
            H_mm=H_mm,
            base_nx=base_nx,
            ny_per_mm=ny_per_mm,
            nz=nz,
            z_grading=z_grading,
        )
    else:
        res = _bm_asymmetric_lumen_3d(
            W_mm=W_mm,
            H_mm=H_mm,
            base_nx=base_nx,
            ny_per_mm=ny_per_mm,
            nz=nz,
            z_grading=z_grading,
        )
    return res


# ---------------------------------------------------------------------------
# Internal block builders
# ---------------------------------------------------------------------------


def _render(
    vertices: Sequence[Tuple[float, float, float]],
    blocks: Sequence[Tuple[Tuple[int, ...], Tuple[int, int, int]]],
    patches: Dict[str, Tuple[str, List[Tuple[int, int, int, int]]]],
) -> str:
    """Render a blockMeshDict from structured inputs."""
    vstr = "\n".join(f"    ({v[0]:.6f} {v[1]:.6f} {v[2]:.6f})" for v in vertices)
    bstr_parts = []
    for verts, (nx, ny, nz) in blocks:
        vs = " ".join(str(v) for v in verts)
        bstr_parts.append(f"    hex ({vs}) ({nx} {ny} {nz}) simpleGrading (1 1 1)")
    bstr = "\n".join(bstr_parts)

    boundary_blocks = []
    for name, (ptype, faces) in patches.items():
        faces_str = "\n".join(
            f"                ({a} {b} {c} {d})" for (a, b, c, d) in faces
        )
        boundary_blocks.append(
            textwrap.dedent(
                f"""\
                {name}
                {{
                    type {ptype};
                    faces
                    (
                {faces_str}
                    );
                }}"""
            )
        )
    boundary_str = "\n".join(boundary_blocks)

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
{vstr}
        );

        blocks
        (
{bstr}
        );

        edges ();

        boundary
        (
{boundary_str}
        );

        mergePatchPairs ();
        """
    )


def _bm_opposing(
    *,
    W_mm: float,
    dz_mm: float,
    base_nx: int,
    ny_per_mm: int,
    delta_W: float,
) -> BlockMeshResult:
    """Three y-strips: lower inlet, middle wall, upper inlet.

    The chamber spans y ∈ [0, W_mm].  Drug inlet is centred at y_drug = W_mm/2
    − delta_W·W_mm, medium inlet at y_med = W_mm/2 + delta_W·W_mm, each with
    width W_INLET_MM.  The four y-interfaces between strips are at
        y0 = 0
        y1 = y_drug − W_INLET_MM/2
        y2 = y_drug + W_INLET_MM/2
        y3 = y_med  − W_INLET_MM/2
        y4 = y_med  + W_INLET_MM/2
        y5 = W_mm
    Five blocks stacked along y, sharing edges; each block's x=0 face gets a
    different patch assignment (inlet_drug / inlet_medium / walls).
    """
    if not 0.05 <= delta_W <= 0.5:
        raise ValueError(f"delta_W must be in [0.05, 0.5]; got {delta_W}")

    # Clamp delta_W so the two inlets (a) do not overlap and (b) fit inside
    # [0, W_mm] with a tiny wall margin. Without this the BO wastes its
    # budget on infeasible geometries (the monotone-y-level check raises).
    half = W_INLET_MM / 2.0
    eps = 0.02 * W_mm
    delta_W_min = (half + eps) / W_mm
    delta_W_max = 0.5 - (half + eps) / W_mm
    if delta_W_max <= delta_W_min:
        raise ValueError(
            f"Chamber W={W_mm} mm is too narrow to host two {W_INLET_MM} mm inlets"
        )
    delta_W_eff = min(max(delta_W, delta_W_min), delta_W_max)

    y_drug = W_mm / 2.0 - delta_W_eff * W_mm
    y_med = W_mm / 2.0 + delta_W_eff * W_mm
    ys = [0.0, y_drug - half, y_drug + half, y_med - half, y_med + half, W_mm]
    if any(ys[i] >= ys[i + 1] for i in range(len(ys) - 1)):
        raise ValueError(
            f"Inlet geometry invalid (W={W_mm} mm, delta_W={delta_W_eff}): y-levels {ys} are not monotone"
        )

    L = L_CHAMBER_MM
    dz = dz_mm

    # 12 vertices per z layer: x ∈ {0, L}, y ∈ ys (6 levels).  Two z layers.
    #
    # Indexing convention: v(ix, iy, iz) with ix ∈ {0,1}, iy ∈ {0..5},
    # iz ∈ {0, 1}.  Global index = iz * 12 + iy * 2 + ix.
    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * dz
        for iy in range(6):
            for ix in (0, 1):
                vertices.append((ix * L, ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 12 + iy * 2 + ix

    blocks = []
    ny_counts = []
    for iy in range(5):
        strip_w = ys[iy + 1] - ys[iy]
        ny = max(2, int(round(strip_w * ny_per_mm)))
        ny_counts.append(ny)
        verts = (
            v(0, iy, 0), v(1, iy, 0), v(1, iy + 1, 0), v(0, iy + 1, 0),
            v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1),
        )
        blocks.append((verts, (base_nx, ny, 1)))

    # Inlet patches: x = 0 face of each strip.  Strips 0, 2, 4 are walls;
    # strip 1 is inlet_drug, strip 3 is inlet_medium.
    x0_face = lambda iy: (v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0))
    x1_face = lambda iy: (v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1))
    ymin_face = lambda iy: (v(0, iy, 0), v(1, iy, 0), v(1, iy, 1), v(0, iy, 1))
    ymax_face = lambda iy: (v(0, iy + 1, 0), v(0, iy + 1, 1), v(1, iy + 1, 1), v(1, iy + 1, 0))
    front_face = lambda iy: (v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0))
    back_face = lambda iy: (v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1))

    patches: Dict[str, Tuple[str, List[Tuple[int, int, int, int]]]] = {
        "inlet_drug": ("patch", [x0_face(1)]),
        "inlet_medium": ("patch", [x0_face(3)]),
        "outlet": ("patch", [x1_face(iy) for iy in range(5)]),
        "walls": ("wall", [x0_face(0), x0_face(2), x0_face(4)]),
        "floor": ("wall", [ymin_face(0)] + [ymax_face(4)]),
        "frontAndBack": (
            "empty",
            [front_face(iy) for iy in range(5)] + [back_face(iy) for iy in range(5)],
        ),
    }

    content = _render(vertices, blocks, patches)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=(W_INLET_MM * 1e-3) * (dz * 1e-3),
        inlet_medium_area_m2=(W_INLET_MM * 1e-3) * (dz * 1e-3),
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=dz * 1e-3,
        base_nx=base_nx,
        base_ny=sum(ny_counts),
    )


def _bm_same_side_y(
    *,
    W_mm: float,
    dz_mm: float,
    base_nx: int,
    ny_per_mm: int,
) -> BlockMeshResult:
    """Two y-stacked blocks; x=0 face splits into inlet_drug (lower) and inlet_medium (upper).

    Downstream of the Y-junction the two feed streams enter the chamber side-
    by-side.  In 2D with a frozen-flow passive-scalar solve this is indistin-
    guishable from two Dirichlet segments on the inlet face, which is what we
    impose here.
    """
    L = L_CHAMBER_MM
    dz = dz_mm
    ys = [0.0, W_mm / 2.0, W_mm]

    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * dz
        for iy in range(3):
            for ix in (0, 1):
                vertices.append((ix * L, ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 6 + iy * 2 + ix

    blocks = []
    ny_counts = []
    for iy in range(2):
        strip_w = ys[iy + 1] - ys[iy]
        ny = max(2, int(round(strip_w * ny_per_mm)))
        ny_counts.append(ny)
        verts = (
            v(0, iy, 0), v(1, iy, 0), v(1, iy + 1, 0), v(0, iy + 1, 0),
            v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1),
        )
        blocks.append((verts, (base_nx, ny, 1)))

    x0_face = lambda iy: (v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0))
    x1_face = lambda iy: (v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1))
    ymin_face = (v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1))
    ymax_face = (v(0, 2, 0), v(0, 2, 1), v(1, 2, 1), v(1, 2, 0))
    front_face = lambda iy: (v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0))
    back_face = lambda iy: (v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1))

    patches = {
        "inlet_drug": ("patch", [x0_face(0)]),
        "inlet_medium": ("patch", [x0_face(1)]),
        "outlet": ("patch", [x1_face(0), x1_face(1)]),
        "walls": ("wall", []),
        "floor": ("wall", [ymin_face, ymax_face]),
        "frontAndBack": (
            "empty",
            [front_face(iy) for iy in range(2)] + [back_face(iy) for iy in range(2)],
        ),
    }

    content = _render(vertices, blocks, patches)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=((W_mm / 2.0) * 1e-3) * (dz * 1e-3),
        inlet_medium_area_m2=((W_mm / 2.0) * 1e-3) * (dz * 1e-3),
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=dz * 1e-3,
        base_nx=base_nx,
        base_ny=sum(ny_counts),
    )


def _bm_asymmetric_lumen(
    *,
    W_mm: float,
    dz_mm: float,
    base_nx: int,
    ny_per_mm: int,
) -> BlockMeshResult:
    """Medium inlet on x=0 (full height); drug inlet on a strip of y=0.

    The drug "lumen" is represented as a permeable band on the y=0 wall
    spanning 60% of the chamber length, centred at x = L/2.  This splits the
    chamber into three x-strips (pre-lumen wall, lumen, post-lumen wall) so
    the lumen face can be named inlet_drug while the remainder of y = 0 stays
    as floor.
    """
    L = L_CHAMBER_MM
    dz = dz_mm
    lumen_len = 0.6 * L
    x0 = 0.5 * (L - lumen_len)
    x1 = 0.5 * (L + lumen_len)
    xs = [0.0, x0, x1, L]
    ys = [0.0, W_mm]

    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * dz
        for iy in range(2):
            for ix in range(4):
                vertices.append((xs[ix], ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        # 8 vertices per z layer: iy*4 + ix
        return iz * 8 + iy * 4 + ix

    blocks = []
    ny = max(4, int(round(W_mm * ny_per_mm)))
    total_nx = 0
    for ix in range(3):
        strip_len = xs[ix + 1] - xs[ix]
        nx = max(4, int(round(strip_len / L * base_nx)))
        total_nx += nx
        verts = (
            v(ix, 0, 0), v(ix + 1, 0, 0), v(ix + 1, 1, 0), v(ix, 1, 0),
            v(ix, 0, 1), v(ix + 1, 0, 1), v(ix + 1, 1, 1), v(ix, 1, 1),
        )
        blocks.append((verts, (nx, ny, 1)))

    ymin_face = lambda ix: (v(ix, 0, 0), v(ix + 1, 0, 0), v(ix + 1, 0, 1), v(ix, 0, 1))
    ymax_face = lambda ix: (v(ix, 1, 0), v(ix, 1, 1), v(ix + 1, 1, 1), v(ix + 1, 1, 0))
    x0_face = (v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0))
    x1_face = (v(3, 0, 0), v(3, 1, 0), v(3, 1, 1), v(3, 0, 1))
    front_face = lambda ix: (v(ix, 0, 0), v(ix, 1, 0), v(ix + 1, 1, 0), v(ix + 1, 0, 0))
    back_face = lambda ix: (v(ix, 0, 1), v(ix + 1, 0, 1), v(ix + 1, 1, 1), v(ix, 1, 1))

    patches = {
        "inlet_medium": ("patch", [x0_face]),
        "outlet": ("patch", [x1_face]),
        "inlet_drug": ("patch", [ymin_face(1)]),  # lumen strip (middle y=0 block)
        "walls": ("wall", []),
        "floor": (
            "wall",
            [ymin_face(0), ymin_face(2), ymax_face(0), ymax_face(1), ymax_face(2)],
        ),
        "frontAndBack": (
            "empty",
            [front_face(ix) for ix in range(3)] + [back_face(ix) for ix in range(3)],
        ),
    }

    content = _render(vertices, blocks, patches)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=(lumen_len * 1e-3) * (dz * 1e-3),
        inlet_medium_area_m2=(W_mm * 1e-3) * (dz * 1e-3),
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=dz * 1e-3,
        base_nx=total_nx,
        base_ny=ny,
    )


# ---------------------------------------------------------------------------
# 3D builders (Module 4.1) — extrude the 2D xy layouts in z with nz layers.
# ---------------------------------------------------------------------------


def _render_3d(
    vertices: Sequence[Tuple[float, float, float]],
    blocks: Sequence[Tuple[Tuple[int, ...], Tuple[int, int, int], Tuple[float, float, float]]],
    patches: Dict[str, Tuple[str, List[Tuple[int, int, int, int]]]],
) -> str:
    """Render a 3D blockMeshDict with per-block simpleGrading in (x, y, z)."""
    vstr = "\n".join(f"    ({v[0]:.6f} {v[1]:.6f} {v[2]:.6f})" for v in vertices)
    bstr_parts = []
    for verts, (nx, ny, nz), (gx, gy, gz) in blocks:
        vs = " ".join(str(v) for v in verts)
        bstr_parts.append(
            f"    hex ({vs}) ({nx} {ny} {nz}) simpleGrading ({gx} {gy} {gz})"
        )
    bstr = "\n".join(bstr_parts)

    boundary_blocks = []
    for name, (ptype, faces) in patches.items():
        faces_str = "\n".join(
            f"                ({a} {b} {c} {d})" for (a, b, c, d) in faces
        )
        boundary_blocks.append(
            textwrap.dedent(
                f"""\
                {name}
                {{
                    type {ptype};
                    faces
                    (
                {faces_str}
                    );
                }}"""
            )
        )
    boundary_str = "\n".join(boundary_blocks)

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
{vstr}
        );

        blocks
        (
{bstr}
        );

        edges ();

        boundary
        (
{boundary_str}
        );

        mergePatchPairs ();
        """
    )


def _bm_opposing_3d(
    *,
    W_mm: float,
    H_mm: float,
    base_nx: int,
    ny_per_mm: int,
    nz: int,
    z_grading: float,
    delta_W: float,
) -> BlockMeshResult:
    """3D extrusion of the 5-strip opposing layout with floor/ceiling patches."""
    if not 0.05 <= delta_W <= 0.5:
        raise ValueError(f"delta_W must be in [0.05, 0.5]; got {delta_W}")

    # See _bm_opposing for rationale: clamp to keep both inlets inside [0, W]
    # and prevent overlap.
    half = W_INLET_MM / 2.0
    eps = 0.02 * W_mm
    delta_W_min = (half + eps) / W_mm
    delta_W_max = 0.5 - (half + eps) / W_mm
    if delta_W_max <= delta_W_min:
        raise ValueError(
            f"Chamber W={W_mm} mm is too narrow to host two {W_INLET_MM} mm inlets"
        )
    delta_W_eff = min(max(delta_W, delta_W_min), delta_W_max)

    y_drug = W_mm / 2.0 - delta_W_eff * W_mm
    y_med = W_mm / 2.0 + delta_W_eff * W_mm
    ys = [0.0, y_drug - half, y_drug + half, y_med - half, y_med + half, W_mm]
    if any(ys[i] >= ys[i + 1] for i in range(len(ys) - 1)):
        raise ValueError(
            f"Inlet geometry invalid (W={W_mm} mm, delta_W={delta_W_eff}): y-levels {ys} are not monotone"
        )

    L = L_CHAMBER_MM
    # 12 vertices per z layer × 2 z layers = 24 vertices.
    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * H_mm
        for iy in range(6):
            for ix in (0, 1):
                vertices.append((ix * L, ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 12 + iy * 2 + ix

    blocks = []
    ny_counts = []
    for iy in range(5):
        strip_w = ys[iy + 1] - ys[iy]
        ny = max(2, int(round(strip_w * ny_per_mm)))
        ny_counts.append(ny)
        verts = (
            v(0, iy, 0), v(1, iy, 0), v(1, iy + 1, 0), v(0, iy + 1, 0),
            v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1),
        )
        blocks.append((verts, (base_nx, ny, nz), (1.0, 1.0, z_grading)))

    x0_face = lambda iy: (v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0))
    x1_face = lambda iy: (v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1))
    ymin_face = lambda iy: (v(0, iy, 0), v(1, iy, 0), v(1, iy, 1), v(0, iy, 1))
    ymax_face = lambda iy: (v(0, iy + 1, 0), v(0, iy + 1, 1), v(1, iy + 1, 1), v(1, iy + 1, 0))
    floor_face = lambda iy: (v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0))
    ceiling_face = lambda iy: (v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1))

    patches: Dict[str, Tuple[str, List[Tuple[int, int, int, int]]]] = {
        "inlet_drug": ("patch", [x0_face(1)]),
        "inlet_medium": ("patch", [x0_face(3)]),
        "outlet": ("patch", [x1_face(iy) for iy in range(5)]),
        "walls": (
            "wall",
            [x0_face(0), x0_face(2), x0_face(4), ymin_face(0), ymax_face(4)],
        ),
        "floor": ("wall", [floor_face(iy) for iy in range(5)]),
        "ceiling": ("wall", [ceiling_face(iy) for iy in range(5)]),
    }

    content = _render_3d(vertices, blocks, patches)
    inlet_area = (W_INLET_MM * 1e-3) * (H_mm * 1e-3)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=inlet_area,
        inlet_medium_area_m2=inlet_area,
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=H_mm * 1e-3,
        base_nx=base_nx,
        base_ny=sum(ny_counts),
    )


def _bm_same_side_y_3d(
    *,
    W_mm: float,
    H_mm: float,
    base_nx: int,
    ny_per_mm: int,
    nz: int,
    z_grading: float,
) -> BlockMeshResult:
    """3D extrusion of the 2-strip Y-junction layout."""
    L = L_CHAMBER_MM
    ys = [0.0, W_mm / 2.0, W_mm]

    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * H_mm
        for iy in range(3):
            for ix in (0, 1):
                vertices.append((ix * L, ys[iy], z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 6 + iy * 2 + ix

    blocks = []
    ny_counts = []
    for iy in range(2):
        strip_w = ys[iy + 1] - ys[iy]
        ny = max(2, int(round(strip_w * ny_per_mm)))
        ny_counts.append(ny)
        verts = (
            v(0, iy, 0), v(1, iy, 0), v(1, iy + 1, 0), v(0, iy + 1, 0),
            v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1),
        )
        blocks.append((verts, (base_nx, ny, nz), (1.0, 1.0, z_grading)))

    x0_face = lambda iy: (v(0, iy, 0), v(0, iy, 1), v(0, iy + 1, 1), v(0, iy + 1, 0))
    x1_face = lambda iy: (v(1, iy, 0), v(1, iy + 1, 0), v(1, iy + 1, 1), v(1, iy, 1))
    ymin_face = (v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1))
    ymax_face = (v(0, 2, 0), v(0, 2, 1), v(1, 2, 1), v(1, 2, 0))
    floor_face = lambda iy: (v(0, iy, 0), v(0, iy + 1, 0), v(1, iy + 1, 0), v(1, iy, 0))
    ceiling_face = lambda iy: (v(0, iy, 1), v(1, iy, 1), v(1, iy + 1, 1), v(0, iy + 1, 1))

    patches = {
        "inlet_drug": ("patch", [x0_face(0)]),
        "inlet_medium": ("patch", [x0_face(1)]),
        "outlet": ("patch", [x1_face(0), x1_face(1)]),
        "walls": ("wall", [ymin_face, ymax_face]),
        "floor": ("wall", [floor_face(0), floor_face(1)]),
        "ceiling": ("wall", [ceiling_face(0), ceiling_face(1)]),
    }

    content = _render_3d(vertices, blocks, patches)
    inlet_area = ((W_mm / 2.0) * 1e-3) * (H_mm * 1e-3)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=inlet_area,
        inlet_medium_area_m2=inlet_area,
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=H_mm * 1e-3,
        base_nx=base_nx,
        base_ny=sum(ny_counts),
    )


def _bm_asymmetric_lumen_3d(
    *,
    W_mm: float,
    H_mm: float,
    base_nx: int,
    ny_per_mm: int,
    nz: int,
    z_grading: float,
) -> BlockMeshResult:
    """3D extrusion of the three x-strip asymmetric-lumen layout."""
    L = L_CHAMBER_MM
    lumen_len = 0.6 * L
    x0 = 0.5 * (L - lumen_len)
    x1 = 0.5 * (L + lumen_len)
    xs = [0.0, x0, x1, L]

    vertices: List[Tuple[float, float, float]] = []
    for iz in (0, 1):
        z = iz * H_mm
        for iy in range(2):
            for ix in range(4):
                vertices.append((xs[ix], iy * W_mm, z))

    def v(ix: int, iy: int, iz: int) -> int:
        return iz * 8 + iy * 4 + ix

    blocks = []
    ny = max(4, int(round(W_mm * ny_per_mm)))
    total_nx = 0
    for ix in range(3):
        strip_len = xs[ix + 1] - xs[ix]
        nx = max(4, int(round(strip_len / L * base_nx)))
        total_nx += nx
        verts = (
            v(ix, 0, 0), v(ix + 1, 0, 0), v(ix + 1, 1, 0), v(ix, 1, 0),
            v(ix, 0, 1), v(ix + 1, 0, 1), v(ix + 1, 1, 1), v(ix, 1, 1),
        )
        blocks.append((verts, (nx, ny, nz), (1.0, 1.0, z_grading)))

    x0_face = (v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0))
    x1_face = (v(3, 0, 0), v(3, 1, 0), v(3, 1, 1), v(3, 0, 1))
    ymin_face = lambda ix: (v(ix, 0, 0), v(ix + 1, 0, 0), v(ix + 1, 0, 1), v(ix, 0, 1))
    ymax_face = lambda ix: (v(ix, 1, 0), v(ix, 1, 1), v(ix + 1, 1, 1), v(ix + 1, 1, 0))
    floor_face = lambda ix: (v(ix, 0, 0), v(ix, 1, 0), v(ix + 1, 1, 0), v(ix + 1, 0, 0))
    ceiling_face = lambda ix: (v(ix, 0, 1), v(ix + 1, 0, 1), v(ix + 1, 1, 1), v(ix, 1, 1))

    patches = {
        "inlet_medium": ("patch", [x0_face]),
        "outlet": ("patch", [x1_face]),
        "inlet_drug": ("patch", [ymin_face(1)]),
        "walls": (
            "wall",
            [ymin_face(0), ymin_face(2), ymax_face(0), ymax_face(1), ymax_face(2)],
        ),
        "floor": ("wall", [floor_face(ix) for ix in range(3)]),
        "ceiling": ("wall", [ceiling_face(ix) for ix in range(3)]),
    }

    content = _render_3d(vertices, blocks, patches)
    return BlockMeshResult(
        content=content,
        inlet_drug_area_m2=(lumen_len * 1e-3) * (H_mm * 1e-3),
        inlet_medium_area_m2=(W_mm * 1e-3) * (H_mm * 1e-3),
        chamber_length_m=L * 1e-3,
        chamber_width_m=W_mm * 1e-3,
        chamber_height_m=H_mm * 1e-3,
        base_nx=total_nx,
        base_ny=ny,
    )
