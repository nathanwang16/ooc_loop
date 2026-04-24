"""Module 1.1 — 1D advection-diffusion analytic solution (no OpenFOAM)."""

from __future__ import annotations

import numpy as np
import pytest

from ooc_optimizer.cfd.scalar import analytic_ad_1d


def test_analytic_ad_boundary_values():
    x = np.array([0.0, 0.5, 1.0]) * 1e-3
    L = 1e-3
    for Pe in (0.1, 1.0, 10.0, 100.0):
        C = analytic_ad_1d(x, L, Pe)
        assert C[0] == pytest.approx(1.0)
        assert C[-1] == pytest.approx(0.0, abs=1e-10)
        assert 0.0 <= C[1] <= 1.0


def test_analytic_ad_diffusion_limit():
    # For Pe → 0 the solution approaches the linear 1 - x/L.
    x = np.linspace(0, 1e-3, 50)
    C = analytic_ad_1d(x, 1e-3, Pe=1e-7)
    assert np.allclose(C, 1.0 - x / 1e-3, atol=1e-3)


def test_analytic_ad_advection_limit():
    # For Pe → ∞ the solution becomes ~1 everywhere except near the outlet.
    x = np.linspace(0, 1e-3, 100)
    C = analytic_ad_1d(x, 1e-3, Pe=1000.0)
    assert C[0] == pytest.approx(1.0)
    assert C[x < 0.9e-3].mean() > 0.95
