"""
Configuration schema loader and validator.

Reads a YAML configuration file and validates all required fields are present
and within acceptable ranges.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "fixed_parameters",
    "continuous_bounds",
    "discrete_levels",
    "solver_settings",
    "paths",
]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file.

    Returns
    -------
    config : dict
        Validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config_path does not exist.
    ValueError
        If required sections or fields are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    logger.info("Configuration loaded from %s", config_path)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Check that all required sections and fields are present."""
    if config is None:
        raise ValueError("Configuration file is empty")

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    _validate_continuous_bounds(config["continuous_bounds"])
    _validate_discrete_levels(config["discrete_levels"])
    _validate_solver_settings(config["solver_settings"])


def _validate_continuous_bounds(bounds: Dict[str, Any]) -> None:
    """Validate that continuous parameter bounds are well-formed."""
    required_params = ["W", "d_p", "s_p", "theta", "Q"]
    for param in required_params:
        if param not in bounds:
            raise ValueError(f"Missing continuous parameter bounds for '{param}'")
        b = bounds[param]
        if "min" not in b or "max" not in b:
            raise ValueError(f"Parameter '{param}' must have 'min' and 'max'")
        if b["min"] >= b["max"]:
            raise ValueError(f"Parameter '{param}': min ({b['min']}) >= max ({b['max']})")


def _validate_discrete_levels(levels: Dict[str, Any]) -> None:
    """Validate discrete parameter level definitions."""
    if "pillar_config" not in levels:
        raise ValueError("Missing 'pillar_config' in discrete_levels")
    if "chamber_height" not in levels:
        raise ValueError("Missing 'chamber_height' in discrete_levels")


def _validate_solver_settings(settings: Dict[str, Any]) -> None:
    """Validate solver configuration fields."""
    required = ["convergence_criterion", "max_iterations", "mesh_resolution"]
    for field in required:
        if field not in settings:
            raise ValueError(f"Missing solver setting: '{field}'")
