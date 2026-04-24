"""
Module 3.1 — Bayesian Optimization Loop

BoTorch-based constrained optimization that calls the CFD evaluation function
to minimize CV(τ) subject to physiological WSS and dead-zone constraints.

Dependencies:
    - cfd module (evaluate_cfd)
    - config module (parameter bounds, settings)

Public API:
    BORunner(config, pillar_config, H) — single-config BO loop
    run_all_configurations(config) — orchestrate all 8 discrete configs
"""

def __getattr__(name):
    if name == "BORunner":
        from ooc_optimizer.optimization.bo_loop import BORunner
        return BORunner
    if name == "run_all_configurations":
        from ooc_optimizer.optimization.orchestrator import run_all_configurations
        return run_all_configurations
    if name == "run_multi_target_workflow":
        from ooc_optimizer.optimization.orchestrator import run_multi_target_workflow
        return run_multi_target_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BORunner",
    "run_all_configurations",
    "run_multi_target_workflow",
]
