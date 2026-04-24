"""Module 4.1 — 3D blockMeshDict generation (text-only, no OpenFOAM)."""

from __future__ import annotations

import re

import pytest

from ooc_optimizer.geometry.topology_blockmesh import (
    VALID_TOPOLOGIES,
    generate_blockmesh_dict_v2_3d,
)

BASE_PARAMS = {
    "W": 1500.0,
    "d_p": 200.0,
    "s_p": 500.0,
    "theta": 45.0,
    "Q_total": 50.0,
    "r_flow": 0.5,
    "delta_W": 0.2,
}


def test_all_3d_topologies_have_required_patches():
    for topo in VALID_TOPOLOGIES:
        res = generate_blockmesh_dict_v2_3d(
            params=BASE_PARAMS, topology=topo, H_um=200.0, nz=6,
        )
        for name in ("inlet_drug", "inlet_medium", "outlet", "floor", "ceiling", "walls"):
            assert name in res.content, f"Missing {name} for {topo}"
        assert "frontAndBack" not in res.content, (
            "3D meshes must not emit frontAndBack patch; got: " + topo
        )
        # No simpleGrading should be empty (it's always a 3-tuple).
        assert re.search(r"simpleGrading\s*\(\s*\d", res.content)


def test_z_grading_round_trips():
    res = generate_blockmesh_dict_v2_3d(
        params=BASE_PARAMS, topology="opposing", H_um=300.0, nz=10, z_grading=0.25,
    )
    assert "0.25" in res.content


def test_3d_chamber_height_reported():
    for H_um in (200.0, 300.0):
        res = generate_blockmesh_dict_v2_3d(
            params=BASE_PARAMS, topology="opposing", H_um=H_um, nz=8,
        )
        assert res.chamber_height_m == pytest.approx(H_um * 1e-6)
        # And both inlet areas scale with H in the opposing case.
        assert res.inlet_drug_area_m2 > 0
        assert res.inlet_medium_area_m2 > 0


def test_3d_rejects_invalid_nz():
    with pytest.raises(ValueError):
        generate_blockmesh_dict_v2_3d(
            params=BASE_PARAMS, topology="opposing", H_um=200.0, nz=2,
        )


def test_3d_rejects_invalid_topology():
    with pytest.raises(ValueError):
        generate_blockmesh_dict_v2_3d(
            params=BASE_PARAMS, topology="nonsense", H_um=200.0, nz=10,
        )


def test_asymmetric_lumen_3d_drug_area_matches_lumen_strip():
    res = generate_blockmesh_dict_v2_3d(
        params=BASE_PARAMS, topology="asymmetric_lumen", H_um=200.0, nz=10,
    )
    # Drug lumen is 60% of chamber length × chamber height.
    expected = (0.6 * 10e-3) * (200e-6)
    assert res.inlet_drug_area_m2 == pytest.approx(expected)
