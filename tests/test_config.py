"""
Tests for Module 2.3 — Configuration Schema and Logging.

Covers:
    - Config loading and validation
    - Evaluation logger append/read operations
    - Best-feasible selection logic
"""

import pytest
import yaml
from pathlib import Path

from ooc_optimizer.config.schema import load_config
from ooc_optimizer.config.logger import EvaluationLogger


def _write_valid_config(path: Path) -> Path:
    """Helper: write a minimal valid config YAML."""
    config = {
        "fixed_parameters": {
            "chamber_length_um": 10000,
            "inlet_width_um": 500,
            "fluid_viscosity_Pa_s": 0.001,
            "fluid_density_kg_m3": 1000,
        },
        "continuous_bounds": {
            "W": {"min": 500, "max": 3000},
            "d_p": {"min": 100, "max": 400},
            "s_p": {"min": 200, "max": 1000},
            "theta": {"min": 15, "max": 75},
            "Q": {"min": 5, "max": 200},
        },
        "discrete_levels": {
            "pillar_config": ["none", "1x4", "2x4", "3x6"],
            "chamber_height": [200, 300],
        },
        "solver_settings": {
            "convergence_criterion": 1e-6,
            "max_iterations": 2000,
            "mesh_resolution": 1,
        },
        "paths": {
            "template_case": "ooc_optimizer/cfd/template",
        },
    }
    config_path = path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestConfigSchema:
    """Configuration loading and validation."""

    def test_load_valid_config(self, tmp_path):
        """A well-formed YAML should load without error."""
        config_path = _write_valid_config(tmp_path)
        config = load_config(config_path)
        assert "fixed_parameters" in config
        assert config["continuous_bounds"]["W"]["min"] == 500

    def test_missing_section_raises(self, tmp_path):
        """Omitting a required section should raise ValueError."""
        config_path = tmp_path / "bad.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"fixed_parameters": {}}, f)
        with pytest.raises(ValueError, match="Missing required"):
            load_config(config_path)

    def test_invalid_bounds_raises(self, tmp_path):
        """min >= max on a continuous param should raise ValueError."""
        config_path = _write_valid_config(tmp_path)
        config = yaml.safe_load(config_path.read_text())
        config["continuous_bounds"]["W"] = {"min": 3000, "max": 500}
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(ValueError, match="min.*>= max"):
            load_config(config_path)

    def test_missing_file_raises(self):
        """Non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_empty_file_raises(self, tmp_path):
        """An empty YAML should raise ValueError."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(config_path)


class TestEvaluationLogger:
    """Structured evaluation logging."""

    def test_log_and_read_back(self, tmp_path):
        """Logged entries should round-trip through JSONL."""
        log = EvaluationLogger(tmp_path / "eval.jsonl")
        log.log_evaluation(
            params={"W": 1500, "Q": 50},
            pillar_config="none",
            H=200,
            metrics={"cv_tau": 0.15, "tau_mean": 1.0, "f_dead": 0.02, "converged": True},
            wall_time_s=12.5,
        )
        records = log.load_all()
        assert len(records) == 1
        assert records[0]["params"]["W"] == 1500
        assert records[0]["metrics"]["cv_tau"] == 0.15

    def test_empty_log_returns_empty(self, tmp_path):
        """Reading a non-existent log file returns empty list."""
        log = EvaluationLogger(tmp_path / "eval.jsonl")
        assert log.load_all() == []

    def test_multiple_entries(self, tmp_path):
        """Multiple entries should all be retrievable."""
        log = EvaluationLogger(tmp_path / "eval.jsonl")
        for i in range(5):
            log.log_evaluation(
                params={"W": 1000 + i * 100},
                pillar_config="none",
                H=200,
                metrics={"cv_tau": 0.5 - i * 0.1, "tau_mean": 1.0, "f_dead": 0.01, "converged": True},
                wall_time_s=10.0,
            )
        assert len(log.load_all()) == 5

    def test_best_feasible_selection(self, tmp_path):
        """Should return the lowest CV(τ) among feasible results."""
        log = EvaluationLogger(tmp_path / "eval.jsonl")

        log.log_evaluation(
            params={"W": 1000}, pillar_config="none", H=200,
            metrics={"cv_tau": 0.3, "tau_mean": 1.0, "f_dead": 0.02, "converged": True},
            wall_time_s=10,
        )
        log.log_evaluation(
            params={"W": 1500}, pillar_config="none", H=200,
            metrics={"cv_tau": 0.1, "tau_mean": 1.0, "f_dead": 0.01, "converged": True},
            wall_time_s=10,
        )
        log.log_evaluation(
            params={"W": 2000}, pillar_config="none", H=200,
            metrics={"cv_tau": 0.05, "tau_mean": 3.0, "f_dead": 0.01, "converged": True},
            wall_time_s=10,
        )

        best = log.get_best_feasible()
        assert best is not None
        assert best["params"]["W"] == 1500

    def test_no_feasible_returns_none(self, tmp_path):
        """If nothing is feasible, return None."""
        log = EvaluationLogger(tmp_path / "eval.jsonl")
        log.log_evaluation(
            params={"W": 1000}, pillar_config="none", H=200,
            metrics={"cv_tau": 999.0, "tau_mean": 0.0, "f_dead": 1.0, "converged": False},
            wall_time_s=10,
        )
        assert log.get_best_feasible() is None
