"""
Validation — Modules 4.1 and 4.3.

v2 3D CFD validation:
    :func:`validate_winner_3d`          run simpleFoam + scalarTransportFoam in
                                        3D for one BO winner.
    :func:`validate_all_winners`        one 3D run per target-profile winner.
    :func:`plot_all_v2`                 the three comparison figures.

v1 WSS premise-test path (retained for the WSS-uniformity exemplar):
    :func:`run_3d_matched_rectangle`    flat-channel 3D sanity case.
    :func:`compare_2d_vs_3d_matched`    v1 profile / scatter / Bland-Altman.
"""

from ooc_optimizer.validation.cfd_3d import (
    compare_2d_vs_3d,
    compare_2d_vs_3d_matched,
    plot_3d_wss_contour,
    plot_streamlines,
    run_3d_matched_rectangle,
    run_3d_validation,
)
from ooc_optimizer.validation.cfd_3d_v2 import (
    Result3D,
    dump_results,
    validate_all_winners,
    validate_winner_3d,
)
from ooc_optimizer.validation.compare_plots_v2 import (
    plot_all_v2,
    plot_centerline_3d_vs_2d,
    plot_concentration_residual_3d_vs_2d,
    plot_wss_scatter_bland_altman,
)

__all__ = [
    # v1 retention
    "run_3d_matched_rectangle",
    "compare_2d_vs_3d_matched",
    "plot_3d_wss_contour",
    "plot_streamlines",
    "run_3d_validation",
    "compare_2d_vs_3d",
    # v2
    "Result3D",
    "validate_winner_3d",
    "validate_all_winners",
    "dump_results",
    "plot_all_v2",
    "plot_centerline_3d_vs_2d",
    "plot_concentration_residual_3d_vs_2d",
    "plot_wss_scatter_bland_altman",
]
