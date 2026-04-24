"""
Module 3.1 — Top-level orchestrator (v2).

For v2 we enumerate 24 discrete configurations (4 pillar layouts × 2 heights ×
3 inlet topologies) and run one BO per configuration on the *primary* target
profile (linear gradient by default).  The winning topology is then re-run
for the remaining target profiles, keeping the overall evaluation budget at
roughly 1,500 forward solves (see Development Guide v2 §2).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ooc_optimizer.optimization.bo_loop import BORunner
from ooc_optimizer.optimization.objectives import TargetProfile, build_target_profile

logger = logging.getLogger(__name__)

PILLAR_CONFIGS = ["none", "1x4", "2x4", "3x6"]
CHAMBER_HEIGHTS = [200.0, 300.0]
INLET_TOPOLOGIES = ["opposing", "same_side_Y", "asymmetric_lumen"]


def _run_single_configuration(
    config: dict, pillar_config: str, H: float, topology: str, target_profile: Optional[TargetProfile]
) -> Dict[str, Any]:
    runner = BORunner(
        config=config,
        pillar_config=pillar_config,
        H=H,
        topology=topology,
        target_profile=target_profile,
    )
    return runner.run()


def run_all_configurations(
    config: dict,
    *,
    parallel: bool = False,
    target_profile: Optional[TargetProfile] = None,
    topologies: Optional[List[str]] = None,
    pillar_configs: Optional[List[str]] = None,
    heights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run BO over a product set of discrete configurations.

    When a subset of ``topologies`` / ``pillar_configs`` / ``heights`` is
    given, only that subset is swept; this is how the top-level workflow
    (``run_multi_target_workflow``) runs just the winning topology for the
    secondary target profiles.
    """
    if config is None:
        raise ValueError("config must not be None")

    # Honor overrides from config["discrete_levels"] when explicit args are
    # not provided.  This lets example configs shrink the sweep (e.g. 15-min
    # smoke run with a single topology/pillar/height).
    levels = config.get("discrete_levels", {}) if isinstance(config, dict) else {}
    topologies = topologies or levels.get("inlet_topology") or INLET_TOPOLOGIES
    pillar_configs = pillar_configs or levels.get("pillar_config") or PILLAR_CONFIGS
    heights = heights or levels.get("chamber_height") or CHAMBER_HEIGHTS
    jobs = [(t, p, h) for t in topologies for p in pillar_configs for h in heights]
    all_runs: List[Dict[str, Any]] = []

    if parallel:
        max_workers = min(len(jobs), 8)
        logger.info("Running %d BO jobs in parallel (workers=%d)", len(jobs), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_run_single_configuration, config, p, h, t, target_profile): (t, p, h)
                for t, p, h in jobs
            }
            for fut in as_completed(futs):
                run_result = fut.result()
                all_runs.append(run_result)
                logger.info(
                    "Completed BO run for %s (%d evaluations)",
                    run_result.get("config_name"), run_result.get("n_evaluations", -1),
                )
    else:
        logger.info("Running %d BO jobs sequentially", len(jobs))
        for t, p, h in jobs:
            run_result = _run_single_configuration(config, p, h, t, target_profile)
            all_runs.append(run_result)

    winner = _select_winner(all_runs)
    return {"winner": winner, "all_runs": all_runs}


def run_multi_target_workflow(
    config: dict,
    *,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Complete v2 workflow: all 24 configs on target 1, then winner-topology on 2 & 3.

    Uses the ``target_profile`` from the config as the primary target.  Two
    extra target profiles (bimodal, step) are appended; these are hard-coded
    because they're the three canonical targets specified in the guide
    (§2). Override ``config["extra_target_profiles"]`` to change.
    """
    primary_spec = dict(config.get("target_profile", {"kind": "linear_gradient"}))
    primary_target = build_target_profile(primary_spec)
    extra_specs = config.get("extra_target_profiles")
    if extra_specs is None:
        extra_specs = [
            {"kind": "bimodal", "peak_axis": "x", "peak_fracs": (0.25, 0.75), "width_frac": 0.1},
            {"kind": "step", "step_axis": "x", "step_frac": 0.5, "sharpness_frac": 0.05},
        ]

    logger.info("Primary target: %s", primary_spec)
    phase1 = run_all_configurations(config, parallel=parallel, target_profile=primary_target)
    winner = phase1.get("winner")
    if winner is None:
        logger.error("No feasible winner for primary target; aborting multi-target workflow")
        return {"primary": phase1, "secondary": []}

    winning_topology = winner["topology"]
    logger.info("Winning topology: %s — running secondary targets on this topology only", winning_topology)

    secondary: List[Dict[str, Any]] = []
    for spec in extra_specs:
        target = build_target_profile(dict(spec))
        result = run_all_configurations(
            config,
            parallel=parallel,
            target_profile=target,
            topologies=[winning_topology],
        )
        result["target_profile"] = spec
        secondary.append(result)

    return {"primary": phase1, "secondary": secondary, "winning_topology": winning_topology}


def _select_winner(all_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the minimum-L2 feasible run across all configurations."""
    feasible = [
        run for run in all_results
        if run.get("best_feasible") is not None
        and run["best_feasible"].get("L2_to_target") is not None
    ]
    if not feasible:
        return None
    best_run = min(feasible, key=lambda r: r["best_feasible"]["L2_to_target"])
    best = best_run["best_feasible"]
    return {
        "config_name": best_run["config_name"],
        "topology": best_run["topology"],
        "pillar_config": best_run["pillar_config"],
        "H": best_run["H"],
        "target_profile": best_run.get("target_profile"),
        "L2_to_target": best["L2_to_target"],
        "params": best["params"],
        "metrics": best["metrics"],
        "state_dir": best_run.get("state_dir"),
    }
