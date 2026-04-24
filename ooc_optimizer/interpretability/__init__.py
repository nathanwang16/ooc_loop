"""
Module 3.3 — Interpretability analysis (v2, NEW).

Primary scientific contribution of v2.  For a BO winner we compute:
    * Sobol indices (global sensitivity) on the trained GP surrogate;
    * GP gradient sensitivities (local, at the optimum);
    * Tolerance intervals per parameter (how far each can drift before the
      L² loss degrades by a configurable amount).
"""

from ooc_optimizer.interpretability.sobol import compute_sobol_indices
from ooc_optimizer.interpretability.gp_gradients import compute_gp_gradients
from ooc_optimizer.interpretability.tolerance import compute_tolerance_intervals
from ooc_optimizer.interpretability.pipeline import analyse_winner, write_heuristic_markdown

__all__ = [
    "compute_sobol_indices",
    "compute_gp_gradients",
    "compute_tolerance_intervals",
    "analyse_winner",
    "write_heuristic_markdown",
]
