"""
Tests for the OpenFOAM ASCII field parser.

Uses synthetic field files — no OpenFOAM installation required.
"""

import numpy as np
import pytest

from ooc_optimizer.cfd.foam_parser import (
    find_latest_time,
    read_scalar_field,
    read_vector_field,
)


# ---------------------------------------------------------------------------
# Scalar field parsing
# ---------------------------------------------------------------------------

class TestReadScalarField:

    def test_nonuniform_scalar(self, tmp_path):
        """Parse a nonuniform scalar field with 5 values."""
        content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar>
5
(
1.0
2.5
3.7
-0.1
4.2e-3
)
;
"""
        field_path = tmp_path / "p"
        field_path.write_text(content)

        values = read_scalar_field(field_path)
        assert values.shape == (5,)
        np.testing.assert_allclose(values, [1.0, 2.5, 3.7, -0.1, 4.2e-3])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_scalar_field(tmp_path / "nonexistent")

    def test_uniform_scalar_raises(self, tmp_path):
        """Uniform fields can't be converted to per-cell arrays."""
        content = """\
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField {}
"""
        field_path = tmp_path / "p"
        field_path.write_text(content)
        with pytest.raises(ValueError, match="uniform"):
            read_scalar_field(field_path)


# ---------------------------------------------------------------------------
# Vector field parsing
# ---------------------------------------------------------------------------

class TestReadVectorField:

    def test_nonuniform_vector(self, tmp_path):
        """Parse a nonuniform vector field with 3 vectors."""
        content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
3
(
(1.0 0.0 0.0)
(0.5 0.1 0.0)
(-0.2 0.3 0.0)
)
;
"""
        field_path = tmp_path / "U"
        field_path.write_text(content)

        values = read_vector_field(field_path)
        assert values.shape == (3, 3)
        np.testing.assert_allclose(values[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(values[1], [0.5, 0.1, 0.0])
        np.testing.assert_allclose(values[2], [-0.2, 0.3, 0.0])

    def test_scientific_notation(self, tmp_path):
        """Parse vectors with scientific notation."""
        content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
2
(
(4.167e-03 0 0)
(6.25e-03 -1.0e-10 0)
)
;
"""
        field_path = tmp_path / "U"
        field_path.write_text(content)

        values = read_vector_field(field_path)
        assert values.shape == (2, 3)
        np.testing.assert_allclose(values[0, 0], 4.167e-03, rtol=1e-10)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_vector_field(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Time directory discovery
# ---------------------------------------------------------------------------

class TestFindLatestTime:

    def test_finds_latest(self, tmp_path):
        """Should return the directory with the highest numeric name."""
        (tmp_path / "0").mkdir()
        (tmp_path / "100").mkdir()
        (tmp_path / "2000").mkdir()
        (tmp_path / "constant").mkdir()
        (tmp_path / "system").mkdir()

        latest = find_latest_time(tmp_path)
        assert latest is not None
        assert latest.name == "2000"

    def test_ignores_zero(self, tmp_path):
        """Time '0' is the initial condition and should be skipped."""
        (tmp_path / "0").mkdir()
        (tmp_path / "constant").mkdir()

        assert find_latest_time(tmp_path) is None

    def test_empty_case_returns_none(self, tmp_path):
        assert find_latest_time(tmp_path) is None

    def test_fractional_times(self, tmp_path):
        """Should handle non-integer time step names."""
        (tmp_path / "0").mkdir()
        (tmp_path / "0.5").mkdir()
        (tmp_path / "1.5").mkdir()

        latest = find_latest_time(tmp_path)
        assert latest.name == "1.5"
