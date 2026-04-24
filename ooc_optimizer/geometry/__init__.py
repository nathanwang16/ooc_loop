"""
Module 1.2 — Parametric Geometry Generator

Produces watertight STL files (fluid domain + mold negative) from a parameter
vector using CadQuery.  Depends only on the config schema for parameter bounds.

Public API:
    generate_chip(params, pillar_config, H, output_dir) -> (fluid_stl, mold_stl)
"""

from ooc_optimizer.geometry.generator import (
    VALID_PILLAR_CONFIGS,
    VALID_TOPOLOGIES,
    generate_chip,
    generate_pillar_obstacles_stl,
)
from ooc_optimizer.geometry.topology_blockmesh import (
    BlockMeshResult,
    generate_blockmesh_dict_v2,
    generate_blockmesh_dict_v2_3d,
)
from ooc_optimizer.geometry.validation import validate_geometry

__all__ = [
    "generate_chip",
    "generate_pillar_obstacles_stl",
    "validate_geometry",
    "generate_blockmesh_dict_v2",
    "generate_blockmesh_dict_v2_3d",
    "BlockMeshResult",
    "VALID_PILLAR_CONFIGS",
    "VALID_TOPOLOGIES",
]
