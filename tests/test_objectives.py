"""Module 2.2 / 3.1 — target profiles and loss functions (pure numerics)."""

from __future__ import annotations

import numpy as np
import pytest

from ooc_optimizer.optimization.objectives import (
    bimodal,
    build_target_profile,
    gradient_sharpness,
    l2_to_target,
    linear_gradient,
    monotonicity_fraction,
    step,
)


L = 10e-3
W = 1.5e-3


def test_linear_gradient_endpoints():
    prof = linear_gradient(axis="x", c_high=1.0, c_low=0.0)
    x = np.array([0.0, L / 2, L])
    y = np.full_like(x, W / 2)
    C = prof.evaluate(x, y, L=L, W=W)
    assert C.shape == (3,)
    assert C[0] == pytest.approx(0.0)
    assert C[1] == pytest.approx(0.5)
    assert C[2] == pytest.approx(1.0)


def test_bimodal_peaks():
    prof = bimodal(peak_axis="x", peak_fracs=(0.25, 0.75), width_frac=0.05, c_peak=1.0, c_base=0.0)
    x = np.array([0.25 * L, 0.5 * L, 0.75 * L])
    y = np.zeros_like(x)
    C = prof.evaluate(x, y, L=L, W=W)
    assert C[0] == pytest.approx(1.0, rel=1e-3)
    assert C[2] == pytest.approx(1.0, rel=1e-3)
    assert C[1] < 0.01


def test_step_sharpness():
    prof = step(step_axis="x", step_frac=0.5, sharpness_frac=0.01, c_high=1.0, c_low=0.0)
    x = np.array([0.3 * L, 0.5 * L, 0.7 * L])
    y = np.zeros_like(x)
    C = prof.evaluate(x, y, L=L, W=W)
    assert C[0] < 0.01
    assert C[1] == pytest.approx(0.5, abs=1e-3)
    assert C[2] > 0.99


def test_build_target_profile_dispatch():
    prof = build_target_profile({"kind": "linear_gradient", "axis": "y", "c_high": 2.0, "c_low": 1.0})
    val = prof.evaluate(np.array([0.0]), np.array([W]), L=L, W=W)
    assert float(val[0]) == pytest.approx(2.0)

    with pytest.raises(ValueError):
        build_target_profile({"kind": "nonexistent"})


def test_l2_to_target_identity_zero():
    C = np.linspace(0, 1, 50)
    assert l2_to_target(C, C) == pytest.approx(0.0, abs=1e-9)


def test_l2_to_target_scales():
    target = np.linspace(0, 1, 50)
    noisy = target + 0.1
    l2 = l2_to_target(noisy, target)
    assert 0.0 < l2 < 1.0


def test_gradient_sharpness_linear_ramp():
    x = np.linspace(0, L, 100)
    C = x / L
    centres = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    # dC/dx = 1/L, so gradient_sharpness = mean(|dC/dx|) * L == 1.0
    gs = gradient_sharpness(C, centres, L=L)
    assert gs == pytest.approx(1.0, rel=0.05)


def test_monotonicity_fraction_perfect():
    x = np.linspace(0, 1, 50)
    C = x.copy()
    centres = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    assert monotonicity_fraction(C, centres, axis="x") == pytest.approx(1.0)


def test_monotonicity_fraction_oscillating():
    x = np.linspace(0, 1, 50)
    C = np.sin(10 * x)
    centres = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    frac = monotonicity_fraction(C, centres, axis="x")
    assert 0.4 < frac < 0.7
