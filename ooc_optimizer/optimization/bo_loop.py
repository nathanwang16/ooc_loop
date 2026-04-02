"""
Bayesian optimization runner for a single discrete configuration.

Uses BoTorch with:
    - Matérn 5/2 GP surrogate
    - 15-point Sobol initialization
    - ConstrainedExpectedImprovement acquisition
    - 35 BO iterations (50 total evaluations)

Constraints:
    c1 = τ_mean - 0.5     (feasible if ≥ 0)
    c2 = 2.0 - τ_mean     (feasible if ≥ 0)
    c3 = 0.05 - f_dead    (feasible if ≥ 0)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

N_SOBOL_INIT = 15
N_BO_ITERATIONS = 35


class BORunner:
    """Bayesian optimization loop for one (pillar_config, H) combination.

    Parameters
    ----------
    config : dict
        Loaded configuration with bounds, solver settings, paths.
    pillar_config : str
        One of {"none", "1x4", "2x4", "3x6"}.
    H : float
        Chamber height in μm.
    """

    def __init__(self, config: dict, pillar_config: str, H: float):
        if config is None:
            raise ValueError("config must not be None")
        if pillar_config not in {"none", "1x4", "2x4", "3x6"}:
            raise ValueError(f"Invalid pillar_config: {pillar_config}")
        if H not in (200.0, 300.0):
            raise ValueError(f"Invalid chamber height: {H}")

        self.config = config
        self.pillar_config = pillar_config
        self.H = H
        self.train_X = None
        self.train_Y = None
        self.train_constraints = None

    def run(self) -> Dict:
        """Execute the full BO loop: Sobol init + BO iterations.

        Returns
        -------
        results : dict
            Contains best_params, best_cv_tau, all evaluations, GP model state.
        """
        raise NotImplementedError("Module 3.1 — BO loop not yet implemented")

    def _generate_sobol_points(self, n: int = N_SOBOL_INIT):
        """Generate quasi-random initial points within parameter bounds."""
        raise NotImplementedError

    def _evaluate_point(self, x) -> Tuple[float, List[float]]:
        """Call the CFD pipeline for one parameter vector.

        Returns (objective, [c1, c2, c3]) tuple.
        """
        raise NotImplementedError

    def _fit_gp(self) -> None:
        """Fit/refit the GP surrogate and constraint GPs."""
        raise NotImplementedError

    def _optimize_acquisition(self):
        """Maximize ConstrainedExpectedImprovement to get next candidate."""
        raise NotImplementedError

    def _get_best_feasible(self) -> Optional[Dict]:
        """Return the best feasible point found so far."""
        raise NotImplementedError

    def save_state(self, output_dir: Path) -> None:
        """Serialize GP model, training data, and evaluation log."""
        raise NotImplementedError
