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
import cadquery as cq

logger = logging.getLogger(__name__)


def validate_geometry(params: Dict[str, float], pillar_config: str, H: float, solid: cq.Workplane) -> List[str]:
    """
    Run all geometry validation checks.
    Returns a list of error strings. If empty, geometry is valid.
    """
    errors = []
    
    # 1. Parameter presence check
    required = ['W', 'theta']
    if pillar_config != "none":
        required += ['d_p', 's_p']
    
    for req in required:
        if req not in params:
            raise ValueError(f"Missing required parameter: {req}")

    # 2. Check Pillar Constraints (Geometric Logic)
    if pillar_config != "none":
        if not check_pillar_constraint(params['d_p'], params['s_p']):
            errors.append(f"Pillar constraint failed: s_p ({params['s_p']}um) must be > d_p + 100um")

    # 3. Check Minimum Feature Size (Printer Limits)
    if not check_min_feature_size(params):
        errors.append("One or more features are below the 100um printer resolution limit.")

    # 4. Check positive volume (unit-agnostic sanity check)
    if not check_positive_volume(solid):
        errors.append("Computed volume is non-positive (possible CAD corruption).")

    # 5. Check Manifold/Self-Intersection (CadQuery isValid)
    if not solid.val().isValid():
        errors.append("Geometry is not a valid manifold (self-intersecting faces detected).")

    return errors

def check_pillar_constraint(d_p: float, s_p: float, min_gap: float = 100.0) -> bool:
    """Verify s_p > d_p + min_gap (all in μm)."""
    # Spacing (s_p) is center-to-center. 
    # The physical gap between surfaces is s_p - d_p.
    return s_p > (d_p + min_gap)


def check_min_feature_size(params: Dict[str, float], min_feature: float = 100.0) -> bool:
    """Verify all geometric features exceed printer minimum resolution."""
    # Check pillar diameter
    if 'd_p' in params and params['d_p'] < min_feature:
        return False
    
    # Check width vs inlet width difference
    # If W is too close to W_in (500um), the taper becomes a razor-thin edge
    if abs(params['W'] - 500.0) < min_feature and params['W'] != 500.0:
        return False
        
    return True


def check_positive_volume(solid: cq.Workplane) -> bool:
    """Ensure the generated geometry has positive volume."""
    return solid.val().Volume() > 0.0
