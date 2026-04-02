"""
Top-level orchestrator for all discrete configurations.

Runs 8 independent BO loops (4 pillar configs × 2 heights), then selects
the overall winner — the lowest CV(τ) among all feasible solutions.
"""

import logging
from typing import Dict, List, Optional

from ooc_optimizer.optimization.bo_loop import BORunner

logger = logging.getLogger(__name__)

PILLAR_CONFIGS = ["none", "1x4", "2x4", "3x6"]
CHAMBER_HEIGHTS = [200.0, 300.0]


def run_all_configurations(
    config: dict,
    parallel: bool = False,
) -> Dict:
    """Execute BO for all 8 discrete configurations and select the winner.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    parallel : bool
        If True, run configurations in parallel (requires multiple cores).

    Returns
    -------
    results : dict
        Keys: 'winner' (best overall), 'all_runs' (per-config results).
    """
    raise NotImplementedError("Module 3.1 — orchestrator not yet implemented")


def _select_winner(all_results: List[Dict]) -> Optional[Dict]:
    """Select the configuration with the lowest feasible CV(τ)."""
    raise NotImplementedError
