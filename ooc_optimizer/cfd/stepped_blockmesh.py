"""
BlockMesh generators for matched and mismatched rectangular OoC chambers.

Matched mode:
    Single block with full-width inlet/outlet.

Mismatched mode:
    Five blocks (inlet stub, three chamber bands, outlet stub) that represent
    a narrow inlet/outlet (W_in) connected to a wide chamber (W_chamber) via
    sudden expansion and contraction.
"""

from __future__ import annotations

import textwrap


def generate_stepped_blockmesh_dict(
    L_chamber_mm: float,
    W_chamber_mm: float,
    W_in_mm: float,
    L_stub_mm: float,
    dz_mm: float,
    nx_chamber: int,
    ny_chamber: int,
    nx_stub: int,
    ny_stub_in: int,
) -> str:
    """Generate blockMeshDict for matched or stepped mismatched geometry."""
    _validate_positive("L_chamber_mm", L_chamber_mm)
    _validate_positive("W_chamber_mm", W_chamber_mm)
    _validate_positive("W_in_mm", W_in_mm)
    _validate_positive("dz_mm", dz_mm)
    _validate_positive("nx_chamber", nx_chamber)
    _validate_positive("ny_chamber", ny_chamber)
    _validate_positive("nx_stub", nx_stub)
    _validate_positive("ny_stub_in", ny_stub_in)
    if W_in_mm > W_chamber_mm:
        raise ValueError("W_in_mm must be <= W_chamber_mm")

    if abs(W_in_mm - W_chamber_mm) < 1e-12:
        return _generate_matched_blockmesh(
            L_total_mm=L_chamber_mm,
            W_mm=W_chamber_mm,
            dz_mm=dz_mm,
            nx=nx_chamber,
            ny=ny_chamber,
        )

    _validate_positive("L_stub_mm", L_stub_mm)
    return _generate_mismatched_blockmesh(
        L_chamber_mm=L_chamber_mm,
        W_chamber_mm=W_chamber_mm,
        W_in_mm=W_in_mm,
        L_stub_mm=L_stub_mm,
        dz_mm=dz_mm,
        nx_chamber=nx_chamber,
        ny_chamber=ny_chamber,
        nx_stub=nx_stub,
        ny_stub_in=ny_stub_in,
    )


def _validate_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def _generate_matched_blockmesh(L_total_mm: float, W_mm: float, dz_mm: float, nx: int, ny: int) -> str:
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
            ({L_total_mm} 0 0)
            ({L_total_mm} {W_mm} 0)
            (0 {W_mm} 0)
            (0 0 {dz_mm})
            ({L_total_mm} 0 {dz_mm})
            ({L_total_mm} {W_mm} {dz_mm})
            (0 {W_mm} {dz_mm})
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
                );
            }}
            floor
            {{
                type wall;
                faces (
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


def _generate_mismatched_blockmesh(
    L_chamber_mm: float,
    W_chamber_mm: float,
    W_in_mm: float,
    L_stub_mm: float,
    dz_mm: float,
    nx_chamber: int,
    ny_chamber: int,
    nx_stub: int,
    ny_stub_in: int,
) -> str:
    side_band = (W_chamber_mm - W_in_mm) / 2.0
    if side_band <= 0:
        raise ValueError("Mismatched mode requires W_chamber_mm > W_in_mm")

    ny_mid = max(1, int(ny_stub_in))
    if ny_chamber <= ny_mid:
        raise ValueError("ny_chamber must be greater than ny_stub_in for mismatched mode")
    ny_side_total = ny_chamber - ny_mid
    ny_bot = max(1, ny_side_total // 2)
    ny_top = max(1, ny_side_total - ny_bot)

    x0 = 0.0
    x1 = L_stub_mm
    x2 = L_stub_mm + L_chamber_mm
    x3 = L_stub_mm + L_chamber_mm + L_stub_mm
    y0 = 0.0
    y1 = side_band
    y2 = side_band + W_in_mm
    y3 = W_chamber_mm

    # 24 vertices: x={x0,x1,x2,x3}; y has {y1,y2} at x0/x3, {y0,y1,y2,y3} at x1/x2.
    # Two z-levels for each 2D point.
    points_2d = [
        (x0, y1), (x0, y2),
        (x1, y0), (x1, y1), (x1, y2), (x1, y3),
        (x2, y0), (x2, y1), (x2, y2), (x2, y3),
        (x3, y1), (x3, y2),
    ]
    vertices: list[tuple[float, float, float]] = []
    index_map: dict[tuple[float, float, int], int] = {}
    for z_idx, z_val in enumerate((0.0, dz_mm)):
        for x_val, y_val in points_2d:
            index_map[(x_val, y_val, z_idx)] = len(vertices)
            vertices.append((x_val, y_val, z_val))

    def v(x_val: float, y_val: float, z_idx: int) -> int:
        return index_map[(x_val, y_val, z_idx)]

    def hex_block(xl: float, xr: float, yb: float, yt: float) -> tuple[int, int, int, int, int, int, int, int]:
        return (
            v(xl, yb, 0),
            v(xr, yb, 0),
            v(xr, yt, 0),
            v(xl, yt, 0),
            v(xl, yb, 1),
            v(xr, yb, 1),
            v(xr, yt, 1),
            v(xl, yt, 1),
        )

    inlet_stub = hex_block(x0, x1, y1, y2)
    chamber_bot = hex_block(x1, x2, y0, y1)
    chamber_mid = hex_block(x1, x2, y1, y2)
    chamber_top = hex_block(x1, x2, y2, y3)
    outlet_stub = hex_block(x2, x3, y1, y2)

    vertices_text = "\n".join(f"        ({x} {y} {z})" for x, y, z in vertices)
    blocks_text = "\n".join(
        [
            f"        hex ({' '.join(map(str, inlet_stub))}) ({nx_stub} {ny_stub_in} 1) simpleGrading (1 1 1)",
            f"        hex ({' '.join(map(str, chamber_bot))}) ({nx_chamber} {ny_bot} 1) simpleGrading (1 1 1)",
            f"        hex ({' '.join(map(str, chamber_mid))}) ({nx_chamber} {ny_mid} 1) simpleGrading (1 1 1)",
            f"        hex ({' '.join(map(str, chamber_top))}) ({nx_chamber} {ny_top} 1) simpleGrading (1 1 1)",
            f"        hex ({' '.join(map(str, outlet_stub))}) ({nx_stub} {ny_stub_in} 1) simpleGrading (1 1 1)",
        ]
    )

    # Patch faces, outward orientation follows the same convention used in the
    # existing verification blockMesh templates.
    inlet_face = f"({v(x0, y1, 0)} {v(x0, y1, 1)} {v(x0, y2, 1)} {v(x0, y2, 0)})"
    outlet_face = f"({v(x3, y2, 0)} {v(x3, y2, 1)} {v(x3, y1, 1)} {v(x3, y1, 0)})"

    walls_faces = [
        f"({v(x0, y1, 0)} {v(x1, y1, 0)} {v(x1, y1, 1)} {v(x0, y1, 1)})",
        f"({v(x0, y2, 0)} {v(x0, y2, 1)} {v(x1, y2, 1)} {v(x1, y2, 0)})",
        f"({v(x2, y1, 0)} {v(x3, y1, 0)} {v(x3, y1, 1)} {v(x2, y1, 1)})",
        f"({v(x2, y2, 0)} {v(x2, y2, 1)} {v(x3, y2, 1)} {v(x3, y2, 0)})",
        f"({v(x1, y0, 0)} {v(x2, y0, 0)} {v(x2, y0, 1)} {v(x1, y0, 1)})",
        f"({v(x1, y1, 0)} {v(x1, y0, 0)} {v(x1, y0, 1)} {v(x1, y1, 1)})",
        f"({v(x2, y0, 0)} {v(x2, y1, 0)} {v(x2, y1, 1)} {v(x2, y0, 1)})",
    ]

    floor_faces = [
        f"({v(x1, y3, 0)} {v(x1, y3, 1)} {v(x2, y3, 1)} {v(x2, y3, 0)})",
        f"({v(x1, y2, 0)} {v(x1, y2, 1)} {v(x1, y3, 1)} {v(x1, y3, 0)})",
        f"({v(x2, y3, 0)} {v(x2, y3, 1)} {v(x2, y2, 1)} {v(x2, y2, 0)})",
    ]

    front_back_faces = [
        f"({a} {d} {c} {b})" for (a, b, c, d, *_rest) in (
            inlet_stub,
            chamber_bot,
            chamber_mid,
            chamber_top,
            outlet_stub,
        )
    ] + [
        f"({e} {f} {g} {h})" for (*_front, e, f, g, h) in (
            inlet_stub,
            chamber_bot,
            chamber_mid,
            chamber_top,
            outlet_stub,
        )
    ]

    walls_text = "\n".join(f"                    {face}" for face in walls_faces)
    floor_text = "\n".join(f"                    {face}" for face in floor_faces)
    front_back_text = "\n".join(f"                    {face}" for face in front_back_faces)

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
{vertices_text}
        );

        blocks
        (
{blocks_text}
        );

        edges ();

        boundary
        (
            inlet
            {{
                type patch;
                faces ({inlet_face});
            }}
            outlet
            {{
                type patch;
                faces ({outlet_face});
            }}
            walls
            {{
                type wall;
                faces
                (
{walls_text}
                );
            }}
            floor
            {{
                type wall;
                faces
                (
{floor_text}
                );
            }}
            frontAndBack
            {{
                type empty;
                faces
                (
{front_back_text}
                );
            }}
        );

        mergePatchPairs ();
        """
    )
