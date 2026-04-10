"""
Top-level orchestrator for all discrete configurations.

Runs 8 independent BO loops (4 pillar configs × 2 heights), then selects
the overall winner — the lowest CV(τ) among all feasible solutions.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

from ooc_optimizer.optimization.bo_loop import BORunner

logger = logging.getLogger(__name__)

PILLAR_CONFIGS = ["none", "1x4", "2x4", "3x6"]
CHAMBER_HEIGHTS = [200.0, 300.0]


def _run_single_configuration(config: dict, pillar_config: str, H: float) -> Dict:
    runner = BORunner(config=config, pillar_config=pillar_config, H=H)
    return runner.run()


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
    if config is None:
        raise ValueError("config must not be None")

    jobs = [(p, h) for p in PILLAR_CONFIGS for h in CHAMBER_HEIGHTS]
    all_runs: List[Dict] = []

    if parallel:
        max_workers = len(jobs)
        logger.info("Running %d BO jobs in parallel (workers=%d)", len(jobs), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_single_configuration, config, pillar, height): (pillar, height)
                for pillar, height in jobs
            }
            for future in as_completed(future_map):
                pillar, height = future_map[future]
                run_result = future.result()
                all_runs.append(run_result)
                logger.info(
                    "Completed BO run for %s, H=%.0f with %d evals",
                    pillar,
                    height,
                    run_result.get("n_evaluations", -1),
                )
    else:
        logger.info("Running %d BO jobs sequentially", len(jobs))
        for pillar, height in jobs:
            run_result = _run_single_configuration(config, pillar, height)
            all_runs.append(run_result)
            logger.info(
                "Completed BO run for %s, H=%.0f with %d evals",
                pillar,
                height,
                run_result.get("n_evaluations", -1),
            )

    winner = _select_winner(all_runs)
    return {"winner": winner, "all_runs": all_runs}


def _select_winner(all_results: List[Dict]) -> Optional[Dict]:
    """Select the configuration with the lowest feasible CV(τ)."""
    feasible_candidates = [
        run for run in all_results
        if run.get("best_feasible") is not None and run["best_feasible"].get("cv_tau") is not None
    ]
    if not feasible_candidates:
        return None

    best_run = min(feasible_candidates, key=lambda r: r["best_feasible"]["cv_tau"])
    best_feasible = best_run["best_feasible"]
    return {
        "config_name": best_run["config_name"],
        "pillar_config": best_run["pillar_config"],
        "H": best_run["H"],
        "cv_tau": best_feasible["cv_tau"],
        "params": best_feasible["params"],
        "metrics": best_feasible["metrics"],
        "state_dir": best_run.get("state_dir"),
    }
