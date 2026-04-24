"""Module 1.2 / 2.1 — topology-aware blockMeshDict generation (text-only)."""

from __future__ import annotations

import pytest

from ooc_optimizer.geometry.topology_blockmesh import (
    VALID_TOPOLOGIES,
    generate_blockmesh_dict_v2,
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


def test_all_topologies_emit_required_patches():
    for topo in VALID_TOPOLOGIES:
        res = generate_blockmesh_dict_v2(params=BASE_PARAMS, topology=topo, H_um=200.0)
        for name in ("inlet_drug", "inlet_medium", "outlet", "floor", "frontAndBack"):
            assert name in res.content, f"Missing patch {name} for topology {topo}"
        assert "convertToMeters 0.001;" in res.content
        assert res.chamber_length_m == pytest.approx(10e-3)


def test_opposing_area_positive():
    res = generate_blockmesh_dict_v2(params=BASE_PARAMS, topology="opposing", H_um=200.0)
    assert res.inlet_drug_area_m2 > 0.0
    assert res.inlet_medium_area_m2 > 0.0
    assert res.inlet_drug_area_m2 == pytest.approx(res.inlet_medium_area_m2)


def test_same_side_y_splits_inlet():
    res = generate_blockmesh_dict_v2(params=BASE_PARAMS, topology="same_side_Y", H_um=200.0)
    # Two inlets share the inlet face, so total area equals chamber-height cross section.
    total = res.inlet_drug_area_m2 + res.inlet_medium_area_m2
    assert total == pytest.approx(res.chamber_width_m * res.chamber_height_m)


def test_asymmetric_lumen_area_different():
    res = generate_blockmesh_dict_v2(params=BASE_PARAMS, topology="asymmetric_lumen", H_um=200.0)
    # The drug inlet is a long-side lumen and the medium inlet is the whole
    # short edge — they should not be equal.
    assert res.inlet_drug_area_m2 != pytest.approx(res.inlet_medium_area_m2)
    assert res.inlet_drug_area_m2 > 0.0


def test_invalid_topology_raises():
    with pytest.raises(ValueError):
        generate_blockmesh_dict_v2(params=BASE_PARAMS, topology="nonsense", H_um=200.0)


def test_delta_W_out_of_range_opposing():
    bad = dict(BASE_PARAMS)
    bad["delta_W"] = 0.6
    with pytest.raises(ValueError):
        generate_blockmesh_dict_v2(params=bad, topology="opposing", H_um=200.0)
