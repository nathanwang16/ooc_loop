"""
Module 3.3 — Interpretability analysis pipeline.

``analyse_winner`` takes a serialised BO run (the ``state_dir`` produced by
Module 3.1) and runs Sobol + GP-gradient + tolerance analyses on it.  A
summary dictionary and a human-readable design-heuristics markdown are
written alongside the figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ooc_optimizer.interpretability.gp_gradients import (
    LocalSensitivity,
    compute_gp_gradients,
    plot_local_sensitivity,
)
from ooc_optimizer.interpretability.sobol import (
    SobolResult,
    compute_sobol_indices,
    plot_sobol_bar,
)
from ooc_optimizer.interpretability.tolerance import (
    ToleranceInterval,
    compute_tolerance_intervals,
    plot_tolerance_intervals,
)

logger = logging.getLogger(__name__)


def _load_run(state_dir: Path) -> Dict[str, Any]:
    state_dir = Path(state_dir)
    evals_path = state_dir / "evaluations.json"
    if not evals_path.exists():
        raise FileNotFoundError(f"Missing BO state file: {evals_path}")
    with open(evals_path, "r") as f:
        return json.load(f)


def _rebuild_gp(run: Dict[str, Any]):
    """Re-instantiate the objective SingleTaskGP from saved training data."""
    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BO stack required for interpretability; install requirements.txt") from exc

    X = torch.tensor(run["train_X"], dtype=torch.double)
    Y = torch.tensor(run["train_Y"], dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _find_optimum_norm(run: Dict[str, Any]) -> np.ndarray:
    """Locate the best-feasible training point, in normalised coords."""
    train_X = np.asarray(run["train_X"], dtype=float)
    train_Y = np.asarray(run["train_Y"], dtype=float)
    constraints = np.asarray(run["train_constraints"], dtype=float)
    feasible = np.all(constraints >= 0, axis=1)
    if feasible.any():
        idx = int(np.argmin(np.where(feasible, train_Y, np.inf)))
    else:
        idx = int(np.argmin(train_Y))
    return train_X[idx]


def analyse_winner(
    state_dir: Path,
    *,
    sobol_n_samples: int = 1024,
    tolerance_loss_tolerance: float = 0.1,
    write_figures: bool = True,
) -> Dict[str, Any]:
    """Run Sobol + GP-gradients + tolerance analyses on a saved BO run.

    Parameters
    ----------
    state_dir : Path
        Directory containing ``evaluations.json`` (produced by Module 3.1).
    sobol_n_samples : int
        Saltelli base sample size.
    tolerance_loss_tolerance : float
        Fractional degradation of the GP mean used to define the tolerance
        interval.
    write_figures : bool
        When True, writes Sobol bar chart, local-sensitivity ranking, and
        tolerance-interval plot to state_dir/interpretability/.

    Returns
    -------
    summary : dict
        Everything downstream consumers need: Sobol table, local sensitivity
        table, tolerance intervals, design-heuristic markdown path.
    """
    state_dir = Path(state_dir)
    run = _load_run(state_dir)
    out_dir = state_dir / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_order = run["parameter_order"]
    active_mask = run["active_mask"]
    active_names = [n for n, a in zip(full_order, active_mask) if a]
    bounds = {name: tuple(run["bounds"][name]) for name in full_order}

    logger.info(
        "Interpretability: %s (active=%s, n_train=%d)",
        state_dir.name, active_names, len(run["train_X"]),
    )

    model = _rebuild_gp(run)
    x_star = _find_optimum_norm(run)

    sobol = compute_sobol_indices(
        model,
        active_names=active_names,
        active_mask=active_mask,
        full_param_order=full_order,
        n_samples=sobol_n_samples,
    )
    grads = compute_gp_gradients(
        model,
        x_optimum_norm=x_star,
        active_names=active_names,
        active_mask=active_mask,
    )
    tolerances = compute_tolerance_intervals(
        model,
        x_optimum_norm=x_star,
        active_names=active_names,
        active_mask=active_mask,
        bounds=bounds,
        loss_tolerance=tolerance_loss_tolerance,
    )

    figs: Dict[str, Optional[str]] = {}
    if write_figures:
        figs["sobol"] = str(plot_sobol_bar(sobol, out_dir / "sobol.png", title=run.get("topology")))
        figs["local_sensitivity"] = str(
            plot_local_sensitivity(grads, out_dir / "local_sensitivity.png", title=run.get("topology"))
        )
        figs["tolerance"] = str(
            plot_tolerance_intervals(tolerances, out_dir / "tolerance.png", title=run.get("topology"))
        )

    summary: Dict[str, Any] = {
        "state_dir": str(state_dir),
        "topology": run.get("topology"),
        "pillar_config": run.get("pillar_config"),
        "H": run.get("H"),
        "target_profile": run.get("target_profile"),
        "x_star_norm": x_star.tolist(),
        "sobol": sobol.to_dict(),
        "local_sensitivity": grads.to_dict(),
        "tolerance_intervals": [iv.to_dict() for iv in tolerances],
        "figures": figs,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    heuristic_path = write_heuristic_markdown(summary, out_dir / "design_heuristics.md")
    summary["heuristic_markdown"] = str(heuristic_path)
    return summary


def write_heuristic_markdown(summary: Dict[str, Any], output_path: Path) -> Path:
    """Translate the numeric interpretability output into a short design note."""
    sobol = summary["sobol"]
    grads = summary["local_sensitivity"]
    intervals: List[Dict[str, Any]] = summary["tolerance_intervals"]

    names = sobol["names"]
    ST = sobol["ST"]
    # "Dominant" parameters: those whose S_T is at least 20% of the max S_T.
    st_max = max(ST) if ST else 1.0
    dominant = [n for n, st in zip(names, ST) if st >= 0.2 * st_max]
    loose = [n for n, st in zip(names, ST) if st < 0.05 * st_max]

    tightest = sorted(
        intervals,
        key=lambda iv: iv["delta_plus_norm"] + iv["delta_minus_norm"],
    )[:3]

    lines = [
        "# Design heuristics",
        "",
        f"**Topology**: `{summary.get('topology')}`  ",
        f"**Pillar config**: `{summary.get('pillar_config')}`  ",
        f"**Chamber height H**: `{summary.get('H')} μm`  ",
        f"**Target profile**: `{summary.get('target_profile')}`",
        "",
        "## Dominant parameters (global sensitivity)",
        "",
    ]
    if dominant:
        lines.append("- " + ", ".join(f"`{n}`" for n in dominant))
    else:
        lines.append("- (none above the 20%·max threshold)")
    lines.extend(["", "## Parameters that can be held loosely", ""])
    if loose:
        lines.append("- " + ", ".join(f"`{n}`" for n in loose))
    else:
        lines.append("- (no parameter with S_T below 5%·max; all matter)")

    lines.extend(
        [
            "",
            "## Local sensitivity ranking (at the BO optimum)",
            "",
            "| Parameter | |∂μ/∂x_norm| |",
            "|---|---|",
        ]
    )
    for name, val in grads["ranking"]:
        lines.append(f"| `{name}` | {val:.4f} |")

    lines.extend(
        [
            "",
            "## Tightest fabrication tolerances",
            "",
            "| Parameter | −Δ (phys) | +Δ (phys) |",
            "|---|---|---|",
        ]
    )
    for iv in tightest:
        lines.append(
            f"| `{iv['name']}` | {iv['delta_minus_phys']:.4g} | {iv['delta_plus_phys']:.4g} |"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
