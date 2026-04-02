"""
Tests for Module 1.1 — Poiseuille Flow Verification.

Split into:
    - Pure-Python tests for analytical formulas (no OpenFOAM required)
    - File generation tests (blockMeshDict, boundary conditions)
    - Integration tests (marked @openfoam, auto-skipped if not installed)
"""

import json
import textwrap

import numpy as np
import pytest

from ooc_optimizer.cfd.verification import (
    PoiseuilleSolution,
    generate_blockmesh_dict,
    generate_inlet_U,
    generate_p_file,
    setup_verification_case,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sol():
    """Standard Poiseuille solution for testing."""
    return PoiseuilleSolution(
        L=10.0e-3,
        W=1.0e-3,
        H=200.0e-6,
        Q_ul_min=50.0,
        mu=1.0e-3,
        rho=1000.0,
    )


# ---------------------------------------------------------------------------
# Analytical formula tests (pure Python — always run)
# ---------------------------------------------------------------------------

class TestPoiseuilleSolution:

    def test_Q_conversion(self, sol):
        """50 μL/min = 8.333e-10 m³/s."""
        expected = 50.0e-9 / 60.0
        assert abs(sol.Q_m3s - expected) / expected < 1e-12

    def test_U_mean(self, sol):
        """U_mean = Q / (W × H)."""
        expected = sol.Q_m3s / (sol.W * sol.H)
        assert abs(sol.U_mean - expected) / expected < 1e-12

    def test_U_centerline(self, sol):
        """U_cl = (3/2) × U_mean for 2D Poiseuille."""
        assert abs(sol.U_centerline / sol.U_mean - 1.5) < 1e-12

    def test_Reynolds_number(self, sol):
        """Re = U_mean × W / ν."""
        Re = sol.U_mean * sol.W / sol.nu
        assert abs(sol.Re - Re) < 1e-12

    def test_kinematic_viscosity(self, sol):
        """ν = μ / ρ = 1e-6 m²/s."""
        assert abs(sol.nu - 1e-6) < 1e-15

    def test_velocity_profile_at_walls(self, sol):
        """u(0) = u(W) = 0 (no-slip)."""
        y = np.array([0.0, sol.W])
        u = sol.velocity_profile(y)
        np.testing.assert_allclose(u, 0.0, atol=1e-15)

    def test_velocity_profile_at_center(self, sol):
        """u(W/2) = U_centerline."""
        u_center = sol.velocity_profile(np.array([sol.W / 2.0]))[0]
        assert abs(u_center - sol.U_centerline) / sol.U_centerline < 1e-12

    def test_velocity_profile_integral(self, sol):
        """∫₀ᵂ u(y) dy / W = U_mean."""
        y = np.linspace(0, sol.W, 10000)
        u = sol.velocity_profile(y)
        U_avg = np.trapezoid(u, y) / sol.W
        assert abs(U_avg - sol.U_mean) / sol.U_mean < 1e-6

    def test_floor_wss_formula(self, sol):
        """τ_floor = 6μu/H at each point."""
        y = np.linspace(0, sol.W, 100)
        u = sol.velocity_profile(y)
        tau_expected = 6.0 * sol.mu * u / sol.H
        tau_computed = sol.floor_wss(y)
        np.testing.assert_allclose(tau_computed, tau_expected, rtol=1e-12)

    def test_floor_wss_mean(self, sol):
        """τ_floor_mean = 6μU_mean/H."""
        expected = 6.0 * sol.mu * sol.U_mean / sol.H
        assert abs(sol.floor_wss_mean - expected) / expected < 1e-12

    def test_floor_wss_centerline(self, sol):
        """τ_floor(W/2) = 9μU_mean/H."""
        expected = 9.0 * sol.mu * sol.U_mean / sol.H
        assert abs(sol.floor_wss_centerline - expected) / expected < 1e-12

    def test_pressure_drop_kinematic(self, sol):
        """Δ(p/ρ) = 12νLU_mean/W²."""
        expected = 12.0 * sol.nu * sol.L * sol.U_mean / (sol.W ** 2)
        assert abs(sol.pressure_drop_kinematic - expected) / expected < 1e-12

    def test_pressure_drop_Pa(self, sol):
        """ΔP = ρ × Δ(p/ρ)."""
        expected = sol.rho * sol.pressure_drop_kinematic
        assert abs(sol.pressure_drop_Pa - expected) / expected < 1e-12

    def test_development_length_short(self, sol):
        """At Re~4.17, L_dev should be < 1 mm (well within 10 mm channel)."""
        assert sol.development_length < 1.0e-3

    def test_symmetry(self, sol):
        """u(y) = u(W-y) — profile is symmetric about centerline."""
        y = np.linspace(0, sol.W, 200)
        u = sol.velocity_profile(y)
        u_mirror = sol.velocity_profile(sol.W - y)
        np.testing.assert_allclose(u, u_mirror, rtol=1e-12)


class TestPoiseuilleSolutionEdgeCases:

    def test_zero_flow_rate(self):
        """Q=0 should give zero velocity everywhere."""
        sol = PoiseuilleSolution(
            L=10e-3, W=1e-3, H=200e-6, Q_ul_min=0.0, mu=1e-3, rho=1000.0,
        )
        assert sol.U_mean == 0.0
        y = np.linspace(0, sol.W, 50)
        np.testing.assert_allclose(sol.velocity_profile(y), 0.0, atol=1e-15)

    def test_narrow_channel(self):
        """W=100μm channel should still compute valid solution."""
        sol = PoiseuilleSolution(
            L=10e-3, W=100e-6, H=200e-6, Q_ul_min=5.0, mu=1e-3, rho=1000.0,
        )
        assert sol.U_centerline == 1.5 * sol.U_mean
        assert sol.Re > 0


# ---------------------------------------------------------------------------
# File generation tests
# ---------------------------------------------------------------------------

class TestBlockMeshDictGeneration:

    def test_generates_valid_content(self):
        """Should produce a parseable blockMeshDict."""
        bmd = generate_blockmesh_dict(10.0, 1.0, 0.01, 200, 20)
        assert "FoamFile" in bmd
        assert "blockMeshDict" in bmd
        assert "convertToMeters 0.001" in bmd
        assert "(200 20 1)" in bmd

    def test_vertex_coordinates(self):
        """Vertices should match specified dimensions."""
        bmd = generate_blockmesh_dict(10.0, 1.0, 0.01, 100, 10)
        assert "10.0" in bmd
        assert "1.0" in bmd
        assert "0.01" in bmd

    def test_boundary_patches(self):
        """All required patches should be present."""
        bmd = generate_blockmesh_dict(10.0, 1.0, 0.01, 100, 10)
        for patch in ("inlet", "outlet", "walls", "frontAndBack"):
            assert patch in bmd

    def test_empty_type_on_frontandback(self):
        """frontAndBack must be type empty for 2D."""
        bmd = generate_blockmesh_dict(10.0, 1.0, 0.01, 100, 10)
        idx = bmd.index("frontAndBack")
        section = bmd[idx:idx + 200]
        assert "type empty" in section

    def test_invalid_dimensions_raises(self):
        with pytest.raises(ValueError):
            generate_blockmesh_dict(-1.0, 1.0, 0.01, 100, 10)

    def test_invalid_cell_count_raises(self):
        with pytest.raises(ValueError):
            generate_blockmesh_dict(10.0, 1.0, 0.01, 0, 10)

    def test_different_resolutions(self):
        """Cell counts should change with resolution."""
        bmd_1x = generate_blockmesh_dict(10.0, 1.0, 0.01, 100, 10)
        bmd_4x = generate_blockmesh_dict(10.0, 1.0, 0.01, 400, 40)
        assert "(100 10 1)" in bmd_1x
        assert "(400 40 1)" in bmd_4x


class TestInletUGeneration:

    def test_contains_fixed_value_inlet(self):
        """Inlet file should contain fixedValue with uniform velocity."""
        u_content = generate_inlet_U(0.004167, 1.0e-3)
        assert "fixedValue" in u_content

    def test_U_mean_appears_in_value(self):
        """The specified U_mean should appear in the inlet value."""
        u_content = generate_inlet_U(0.004167, 1.0e-3)
        assert "0.004167" in u_content

    def test_boundary_patches_present(self):
        """All required patches in U file."""
        u_content = generate_inlet_U(0.004167, 1.0e-3)
        for patch in ("inlet", "outlet", "walls", "frontAndBack"):
            assert patch in u_content


class TestPFileGeneration:

    def test_outlet_fixed_value(self):
        """Outlet should be fixedValue 0."""
        p_content = generate_p_file()
        assert "fixedValue" in p_content
        assert "uniform 0" in p_content

    def test_inlet_zero_gradient(self):
        """Inlet should be zeroGradient."""
        p_content = generate_p_file()
        idx = p_content.index("inlet")
        section = p_content[idx:idx + 100]
        assert "zeroGradient" in section


# ---------------------------------------------------------------------------
# Case setup test (filesystem only, no OpenFOAM)
# ---------------------------------------------------------------------------

class TestVerificationCaseSetup:

    def test_creates_case_directory(self, sol, template_case_dir, tmp_path):
        """setup_verification_case should create a complete case directory."""
        case_dir = tmp_path / "test_case"
        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_case_dir,
            sol=sol,
            nx=50,
            ny=5,
        )
        assert (case_dir / "system" / "blockMeshDict").exists()
        assert (case_dir / "system" / "controlDict").exists()
        assert (case_dir / "system" / "fvSchemes").exists()
        assert (case_dir / "system" / "fvSolution").exists()
        assert (case_dir / "constant" / "transportProperties").exists()
        assert (case_dir / "0" / "U").exists()
        assert (case_dir / "0" / "p").exists()

    def test_blockmesh_uses_correct_dimensions(self, sol, template_case_dir, tmp_path):
        """blockMeshDict should have the channel dimensions from the solution."""
        case_dir = tmp_path / "test_case"
        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_case_dir,
            sol=sol,
            nx=100,
            ny=10,
        )
        bmd = (case_dir / "system" / "blockMeshDict").read_text()
        L_mm = sol.L * 1000
        W_mm = sol.W * 1000
        assert f"{L_mm}" in bmd
        assert f"{W_mm}" in bmd

    def test_inlet_has_correct_U_mean(self, sol, template_case_dir, tmp_path):
        """The U file should contain the computed mean velocity."""
        case_dir = tmp_path / "test_case"
        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_case_dir,
            sol=sol,
            nx=50,
            ny=5,
        )
        u_content = (case_dir / "0" / "U").read_text()
        assert "fixedValue" in u_content
        assert str(sol.U_mean)[:6] in u_content

    def test_missing_template_raises(self, sol, tmp_path):
        """Non-existent template should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            setup_verification_case(
                case_dir=tmp_path / "case",
                template_dir=tmp_path / "nonexistent",
                sol=sol,
                nx=50,
                ny=5,
            )


# ---------------------------------------------------------------------------
# OpenFOAM integration tests (auto-skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.openfoam
@pytest.mark.slow
class TestOpenFOAMIntegration:

    def test_poiseuille_verification_passes(self, sol, template_case_dir, tmp_path):
        """Full end-to-end: blockMesh + simpleFoam + comparison < 2% error."""
        from ooc_optimizer.cfd.verification import (
            extract_verification_results,
            run_openfoam_case,
        )

        case_dir = tmp_path / "integration_test"
        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_case_dir,
            sol=sol,
            nx=200,
            ny=20,
        )

        converged = run_openfoam_case(case_dir, timeout_s=120)
        assert converged, "simpleFoam did not converge"

        results = extract_verification_results(case_dir, sol)
        assert results["passed_2pct"], (
            f"Verification failed: "
            f"U_cl_err={results['centerline_velocity_error']:.4%}, "
            f"τ_mean_err={results['tau_mean_error']:.4%}"
        )

    def test_mesh_convergence_study(self, sol, template_case_dir, tmp_path):
        """Convergence study should show < 2% τ_mean change between 2× and 4×."""
        from ooc_optimizer.cfd.verification import run_mesh_convergence

        results = run_mesh_convergence(
            template_dir=template_case_dir,
            output_dir=tmp_path / "convergence",
            sol=sol,
            refinement_levels=[1, 2, 4],
        )

        converged_results = [r for r in results if r.get("converged", False)]
        assert len(converged_results) >= 2, "Need at least 2 converged levels"

        last = converged_results[-1]
        assert last.get("tau_mean_relative_change", 1.0) < 0.02, (
            f"Mesh not converged: τ_mean change = "
            f"{last.get('tau_mean_relative_change', 'N/A'):.4%}"
        )
