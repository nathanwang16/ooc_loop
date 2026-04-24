import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ooc_optimizer.cfd.stepped_blockmesh import generate_stepped_blockmesh_dict


def _extract_vertices_section(blockmesh_text: str) -> str:
    match = re.search(r"vertices\s*\((.*?)\);\s*blocks", blockmesh_text, flags=re.DOTALL)
    if match is None:
        raise AssertionError("Failed to parse vertices section")
    return match.group(1)


def _count_vertices(blockmesh_text: str) -> int:
    vertices_section = _extract_vertices_section(blockmesh_text)
    return len(re.findall(r"\([^\(\)]*\)", vertices_section))


def _extract_patch_faces(blockmesh_text: str, patch_name: str) -> list[tuple[int, int, int, int]]:
    patch_match = re.search(
        rf"{re.escape(patch_name)}\s*\{{.*?faces\s*(\([^\)]*\)[^;]*?)\s*;\s*\}}",
        blockmesh_text,
        flags=re.DOTALL,
    )
    if patch_match is None:
        raise AssertionError(f"Failed to parse patch '{patch_name}'")
    face_tuples = re.findall(r"\((\d+)\s+(\d+)\s+(\d+)\s+(\d+)\)", patch_match.group(1))
    return [tuple(map(int, face)) for face in face_tuples]


def _extract_vertices_list(blockmesh_text: str) -> list[tuple[float, float, float]]:
    vertices = []
    for triplet in re.findall(r"\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", _extract_vertices_section(blockmesh_text)):
        vertices.append((float(triplet[0]), float(triplet[1]), float(triplet[2])))
    return vertices


def _face_area_in_model_units(face: tuple[int, int, int, int], vertices: list[tuple[float, float, float]]) -> float:
    p0 = vertices[face[0]]
    p1 = vertices[face[1]]
    p2 = vertices[face[2]]
    p3 = vertices[face[3]]
    # For inlet/outlet faces in these meshes, x is constant and this reduces to
    # projected rectangle area in y-z.
    ys = [p0[1], p1[1], p2[1], p3[1]]
    zs = [p0[2], p1[2], p2[2], p3[2]]
    return (max(ys) - min(ys)) * (max(zs) - min(zs))


def test_matched_mode_collapses_to_single_block() -> None:
    content = generate_stepped_blockmesh_dict(
        L_chamber_mm=10.0,
        W_chamber_mm=2.5,
        W_in_mm=2.5,
        L_stub_mm=2.0,
        dz_mm=0.01,
        nx_chamber=200,
        ny_chamber=50,
        nx_stub=40,
        ny_stub_in=20,
    )
    assert _count_vertices(content) == 8
    assert "hex (0 1 2 3 4 5 6 7) (200 50 1)" in content
    assert "frontAndBack" in content
    assert "type empty" in content


def test_mismatched_mode_uses_expected_vertex_count() -> None:
    content = generate_stepped_blockmesh_dict(
        L_chamber_mm=10.0,
        W_chamber_mm=3.0,
        W_in_mm=0.5,
        L_stub_mm=2.0,
        dz_mm=0.01,
        nx_chamber=200,
        ny_chamber=60,
        nx_stub=40,
        ny_stub_in=10,
    )
    assert _count_vertices(content) == 24
    assert content.count("hex (") == 5


def test_inlet_patch_area_matches_narrow_width() -> None:
    w_in = 0.5
    dz = 0.01
    content = generate_stepped_blockmesh_dict(
        L_chamber_mm=10.0,
        W_chamber_mm=3.0,
        W_in_mm=w_in,
        L_stub_mm=2.0,
        dz_mm=dz,
        nx_chamber=200,
        ny_chamber=60,
        nx_stub=40,
        ny_stub_in=10,
    )
    vertices = _extract_vertices_list(content)
    inlet_faces = _extract_patch_faces(content, "inlet")
    assert len(inlet_faces) == 1
    area = _face_area_in_model_units(inlet_faces[0], vertices)
    assert area == pytest.approx(w_in * dz, rel=1e-10)


def test_invalid_width_raises() -> None:
    with pytest.raises(ValueError, match="W_in_mm must be <= W_chamber_mm"):
        generate_stepped_blockmesh_dict(
            L_chamber_mm=10.0,
            W_chamber_mm=0.5,
            W_in_mm=0.8,
            L_stub_mm=2.0,
            dz_mm=0.01,
            nx_chamber=200,
            ny_chamber=40,
            nx_stub=40,
            ny_stub_in=10,
        )


@pytest.mark.openfoam
def test_generated_mismatched_mesh_passes_checkmesh(tmp_path: Path) -> None:
    case_dir = tmp_path / "mismatched_case"
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    template_system = repo_root / "ooc_optimizer" / "cfd" / "template" / "system"
    template_constant = repo_root / "ooc_optimizer" / "cfd" / "template" / "constant"
    (case_dir / "system" / "controlDict").write_text(
        (template_system / "controlDict").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSchemes").write_text(
        (template_system / "fvSchemes").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSolution").write_text(
        (template_system / "fvSolution").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (case_dir / "constant" / "transportProperties").write_text(
        (template_constant / "transportProperties").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (case_dir / "constant" / "turbulenceProperties").write_text(
        (template_constant / "turbulenceProperties").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    bmd = generate_stepped_blockmesh_dict(
        L_chamber_mm=10.0,
        W_chamber_mm=2.5,
        W_in_mm=0.5,
        L_stub_mm=2.0,
        dz_mm=0.01,
        nx_chamber=120,
        ny_chamber=50,
        nx_stub=30,
        ny_stub_in=10,
    )
    (case_dir / "system" / "blockMeshDict").write_text(bmd, encoding="utf-8")

    if shutil.which("blockMesh") is not None:
        block_cmd = ["blockMesh", "-case", str(case_dir)]
        check_cmd = ["checkMesh", "-case", str(case_dir), "-allGeometry", "-allTopology"]
    else:
        wrapper = None
        for candidate in ("openfoam2406", "openfoam2412", "openfoam2506", "openfoam2512"):
            if shutil.which(candidate):
                wrapper = candidate
                break
        if wrapper is None:
            pytest.skip("OpenFOAM wrapper not found")
        block_cmd = [wrapper, "-c", f"blockMesh -case {case_dir}"]
        check_cmd = [wrapper, "-c", f"checkMesh -case {case_dir} -allGeometry -allTopology"]

    block = subprocess.run(block_cmd, capture_output=True, text=True, check=False)
    assert block.returncode == 0, block.stderr
    check = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr
    assert ("Mesh OK" in check.stdout) or ("Failed 0 mesh checks" in check.stdout)
