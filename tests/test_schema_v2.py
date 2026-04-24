"""Module 2.3 — pydantic schema validation (v2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ooc_optimizer.config.schema import (
    OocConfigV2,
    _normalize_legacy_fields,
    load_config,
)


def _minimal_v2_dict() -> dict:
    return {
        "fixed_parameters": {
            "chamber_length_um": 10000,
            "inlet_width_um": 500,
            "fluid_viscosity_Pa_s": 1e-3,
            "fluid_density_kg_m3": 1000,
        },
        "continuous_bounds": {
            "W": {"min": 500, "max": 3000},
            "d_p": {"min": 100, "max": 400},
            "s_p": {"min": 200, "max": 1000},
            "theta": {"min": 15, "max": 75},
            "Q_total": {"min": 5, "max": 200},
            "r_flow": {"min": 0.1, "max": 0.9},
            "delta_W": {"min": 0.1, "max": 0.45},
        },
        "discrete_levels": {
            "pillar_config": ["none", "1x4", "2x4", "3x6"],
            "chamber_height": [200, 300],
            "inlet_topology": ["opposing", "same_side_Y", "asymmetric_lumen"],
        },
        "solver_settings": {
            "convergence_criterion": 1e-6,
            "max_iterations": 2000,
            "mesh_resolution": 1,
        },
        "paths": {
            "template_case": "ooc_optimizer/cfd/template",
            "stl_output_dir": "data/stl",
            "case_output_dir": "data/cases",
            "results_dir": "data/results",
            "evaluation_log": "data/results/evaluations.jsonl",
            "figures_dir": "figures",
        },
    }


def test_minimal_config_validates():
    model = OocConfigV2.model_validate(_minimal_v2_dict())
    assert model.target_profile.kind == "linear_gradient"
    assert model.optimization.n_sobol_init == 20


def test_bad_bounds_rejected():
    bad = _minimal_v2_dict()
    bad["continuous_bounds"]["W"] = {"min": 100, "max": 50}
    with pytest.raises(ValueError):
        OocConfigV2.model_validate(bad)


def test_target_profile_discriminated_union():
    cfg = _minimal_v2_dict()
    cfg["target_profile"] = {"kind": "bimodal", "peak_fracs": [0.2, 0.8]}
    model = OocConfigV2.model_validate(cfg)
    assert model.target_profile.kind == "bimodal"
    assert tuple(model.target_profile.peak_fracs) == (0.2, 0.8)


def test_v1_legacy_normalisation():
    v1 = _minimal_v2_dict()
    v1["continuous_bounds"].pop("Q_total")
    v1["continuous_bounds"]["Q"] = {"min": 5, "max": 200}
    v1["continuous_bounds"].pop("r_flow")
    v1["continuous_bounds"].pop("delta_W")
    v1["discrete_levels"].pop("inlet_topology")

    normalised = _normalize_legacy_fields(v1)
    assert "Q_total" in normalised["continuous_bounds"]
    assert "r_flow" in normalised["continuous_bounds"]
    assert "delta_W" in normalised["continuous_bounds"]
    assert "inlet_topology" in normalised["discrete_levels"]
    # And the normalised dict validates cleanly.
    OocConfigV2.model_validate(normalised)


def test_default_config_yaml_loads(tmp_path):
    root = Path(__file__).resolve().parent.parent
    raw = (root / "configs" / "default_config.yaml").read_text()
    tmp_path = tmp_path / "copy.yaml"
    tmp_path.write_text(raw)
    cfg = load_config(tmp_path)
    assert "continuous_bounds" in cfg
    assert cfg["continuous_bounds"]["r_flow"]["min"] == 0.1
    assert cfg["target_profile"]["kind"] == "linear_gradient"
