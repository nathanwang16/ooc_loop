"""
Geometry validation checks run automatically on every generated geometry.

Checks:
    - Volume within expected range for given parameters
    - No self-intersecting faces (CadQuery isValid())
    - All features exceed printer minimum of 100 μm
    - Pillar constraint s_p > d_p + 100 μm satisfied
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def validate_geometry(params: Dict[str, float], pillar_config: str) -> List[str]:
    """Run all geometry validation checks.

    Returns
    -------
    errors : list of str
        Empty list if all checks pass; otherwise each string describes a violation.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    raise NotImplementedError("Module 1.2 — geometry validation not yet implemented")


def check_pillar_constraint(d_p: float, s_p: float, min_gap: float = 100.0) -> bool:
    """Verify s_p > d_p + min_gap (all in μm)."""
    raise NotImplementedError


def check_min_feature_size(params: Dict[str, float], min_feature: float = 100.0) -> bool:
    """Verify all geometric features exceed printer minimum resolution."""
    raise NotImplementedError


def check_volume_range(solid, params: Dict[str, float], H: float) -> bool:
    """Verify computed volume is within expected bounds for the parameter set."""
    raise NotImplementedError
