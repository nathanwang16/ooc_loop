"""
Module 1.2 — Parametric Geometry Generator

Produces watertight STL files (fluid domain + mold negative) from a parameter
vector using CadQuery.  Depends only on the config schema for parameter bounds.

Public API:
    generate_chip(params, pillar_config, H, output_dir) -> (fluid_stl, mold_stl)
"""

from ooc_optimizer.geometry.generator import generate_chip
from ooc_optimizer.geometry.validation import validate_geometry

__all__ = ["generate_chip", "validate_geometry"]
