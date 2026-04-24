"""
Module 3.2 — Results Analysis and Visualization (v2).

Public helpers for concentration-field plots, BO convergence curves, and
winner grids.  WSS-centric v1 plots (``wss_contours``, ``comparison``,
``convergence``) are preserved for the retained WSS-uniformity exemplar
and are accessed via explicit imports.
"""

from ooc_optimizer.analysis.concentration_fields import (
    plot_bo_convergence,
    plot_centerline_profile,
    plot_concentration_contour,
    plot_residual_field,
    plot_streamline_overlay,
    plot_winner_grid,
)

__all__ = [
    "plot_bo_convergence",
    "plot_centerline_profile",
    "plot_concentration_contour",
    "plot_residual_field",
    "plot_streamline_overlay",
    "plot_winner_grid",
]
