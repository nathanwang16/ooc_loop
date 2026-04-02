"""
Tests for Module 2.2 — CFD Run Automation and Metric Extraction.

Covers:
    - Full evaluate_cfd pipeline (integration test, requires OpenFOAM)
    - Metric computation from known velocity fields
    - Penalty return on failure
"""

import pytest


class TestMetricExtraction:
    """WSS and flow metric computation."""

    def test_floor_wss_formula(self):
        """τ = 6μU/H on a known uniform field should match analytically."""
        raise NotImplementedError

    def test_dead_fraction_zero_for_uniform_flow(self):
        """Perfectly uniform flow should have f_dead = 0."""
        raise NotImplementedError

    def test_cv_tau_zero_for_uniform_wss(self):
        """Uniform WSS field should yield CV(τ) = 0."""
        raise NotImplementedError


class TestCFDEvaluation:
    """End-to-end CFD evaluation (integration, requires OpenFOAM)."""

    @pytest.mark.slow
    def test_penalty_on_bad_geometry(self):
        """Invalid geometry should return penalty metrics, not crash."""
        raise NotImplementedError
