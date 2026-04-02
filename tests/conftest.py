"""
Shared pytest fixtures and configuration.

Markers:
    @pytest.mark.slow       — requires significant runtime (CFD simulations)
    @pytest.mark.openfoam   — requires a working OpenFOAM installation
"""

import shutil
import subprocess

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root():
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture
def template_case_dir():
    """Path to the OpenFOAM template case."""
    return PROJECT_ROOT / "ooc_optimizer" / "cfd" / "template"


@pytest.fixture
def default_config_path():
    """Path to the default YAML configuration file."""
    return PROJECT_ROOT / "configs" / "default_config.yaml"


@pytest.fixture
def verification_params():
    """Standard Poiseuille verification parameters (SI units)."""
    return {
        "L": 10.0e-3,       # channel length, m
        "W": 1.0e-3,        # channel width, m
        "H": 200.0e-6,      # channel height, m
        "Q_ul_min": 50.0,   # flow rate, μL/min
        "mu": 1.0e-3,       # dynamic viscosity, Pa·s
        "rho": 1000.0,      # density, kg/m³
        "nu": 1.0e-6,       # kinematic viscosity, m²/s
    }


def _openfoam_available() -> bool:
    """Check whether OpenFOAM is available (native or via wrapper)."""
    if shutil.which("simpleFoam") is not None:
        return True
    for wrapper in ("openfoam2406", "openfoam2412", "openfoam2506", "openfoam2512"):
        if shutil.which(wrapper) is not None:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked @openfoam when OpenFOAM is not installed."""
    if _openfoam_available():
        return
    skip_foam = pytest.mark.skip(reason="OpenFOAM not found on PATH")
    for item in items:
        if "openfoam" in item.keywords:
            item.add_marker(skip_foam)
