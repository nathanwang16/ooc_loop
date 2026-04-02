"""
CFD Engine — Modules 1.1, 2.1, 2.2

Wraps OpenFOAM mesh generation, solver execution, and metric extraction into
a single callable evaluation function.

Dependencies:
    - geometry module (provides STL input)
    - config module (solver settings, paths)

Public API:
    evaluate_cfd(params, pillar_config, H, config) -> metrics_dict
"""

from ooc_optimizer.cfd.solver import evaluate_cfd

__all__ = ["evaluate_cfd"]
