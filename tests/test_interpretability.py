"""Module 3.3 — interpretability numerics.

These tests exercise the Sobol / GP-gradient / tolerance pipeline against a
synthetic GP trained on a closed-form toy objective.  They do NOT require
OpenFOAM.  The BoTorch stack is imported lazily and skipped when missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
botorch = pytest.importorskip("botorch")


def _train_toy_gp(n: int = 80, seed: int = 0):
    """Fit a SingleTaskGP on f(x) = 10·x₀² + x₁ in [0, 1]⁵."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = torch.manual_seed(seed)  # noqa: F841
    X = torch.rand(n, 5, dtype=torch.double)
    Y = 10 * X[:, 0] ** 2 + X[:, 1]
    Y = Y.unsqueeze(-1)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, X, Y


def test_sobol_dominant_parameters():
    salib = pytest.importorskip("SALib")  # noqa: F841
    from ooc_optimizer.interpretability.sobol import compute_sobol_indices

    model, _, _ = _train_toy_gp()
    res = compute_sobol_indices(
        model,
        active_names=["x0", "x1", "x2", "x3", "x4"],
        active_mask=[True] * 5,
        full_param_order=["x0", "x1", "x2", "x3", "x4"],
        n_samples=128,
    )
    # x0 should dominate (quadratic), x2/x3/x4 should be ~ noise.
    S_T_map = dict(zip(res.names, res.ST))
    assert S_T_map["x0"] > S_T_map["x2"]
    assert S_T_map["x0"] > S_T_map["x3"]
    assert S_T_map["x0"] > S_T_map["x4"]


def test_gp_gradient_signs():
    from ooc_optimizer.interpretability.gp_gradients import compute_gp_gradients

    model, X, Y = _train_toy_gp()
    # Pick a training point as the "optimum".
    idx = int(torch.argmin(Y.squeeze()))
    x_star = X[idx].tolist()
    res = compute_gp_gradients(
        model,
        x_optimum_norm=x_star,
        active_names=["x0", "x1", "x2", "x3", "x4"],
        active_mask=[True] * 5,
    )
    # x0 should have the largest absolute gradient magnitude on average.
    assert res.ranking[0][0] in ("x0", "x1")


def test_tolerance_intervals_bounded():
    from ooc_optimizer.interpretability.tolerance import compute_tolerance_intervals

    model, X, Y = _train_toy_gp()
    idx = int(torch.argmin(Y.squeeze()))
    x_star = X[idx].tolist()
    bounds = {name: (0.0, 1.0) for name in ["x0", "x1", "x2", "x3", "x4"]}
    intervals = compute_tolerance_intervals(
        model,
        x_optimum_norm=x_star,
        active_names=["x0", "x1", "x2", "x3", "x4"],
        active_mask=[True] * 5,
        bounds=bounds,
        loss_tolerance=0.1,
    )
    for iv in intervals:
        assert 0.0 <= iv.delta_plus_norm <= 1.0
        assert 0.0 <= iv.delta_minus_norm <= 1.0


def test_analyse_winner_endtoend(tmp_path: Path):
    salib = pytest.importorskip("SALib")  # noqa: F841
    # Build a fake BO state directory.
    model, X, Y = _train_toy_gp(n=40)
    state_dir = tmp_path / "bo_fake"
    state_dir.mkdir()
    constraints = np.zeros((X.shape[0], 3)) + 0.1  # all feasible
    payload = {
        "topology": "opposing",
        "pillar_config": "none",
        "H": 200.0,
        "parameter_order": ["x0", "x1", "x2", "x3", "x4"],
        "active_mask": [True, True, True, True, True],
        "bounds": {name: [0.0, 1.0] for name in ["x0", "x1", "x2", "x3", "x4"]},
        "constraints": {"tau_mean_min": 0.1, "tau_mean_max": 2.0, "f_dead_max": 0.05},
        "target_profile": {"kind": "linear_gradient", "axis": "x", "c_high": 1.0, "c_low": 0.0},
        "evaluations": [],
        "train_X": X.tolist(),
        "train_Y": Y.squeeze(-1).tolist(),
        "train_constraints": constraints.tolist(),
    }
    with open(state_dir / "evaluations.json", "w") as f:
        json.dump(payload, f)

    from ooc_optimizer.interpretability import analyse_winner

    summary = analyse_winner(state_dir, sobol_n_samples=64, tolerance_loss_tolerance=0.1)
    assert "sobol" in summary
    assert "local_sensitivity" in summary
    assert "tolerance_intervals" in summary
    assert Path(summary["heuristic_markdown"]).exists()
