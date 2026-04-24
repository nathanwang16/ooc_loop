"""
Module 3.3 — Global sensitivity via Sobol indices on the trained GP surrogate.

SALib is used to generate the saltelli sample and attribute variance to each
input parameter.  The surrogate (BoTorch SingleTaskGP from Module 3.1) is
evaluated at the sample points; we use the GP posterior mean.

Outputs
-------
``compute_sobol_indices`` returns a dict with:
    names         : list of parameter names (masked-out entries excluded);
    S1            : first-order indices;
    S1_conf       : bootstrap CI half-widths;
    ST            : total-order indices;
    ST_conf       : bootstrap CI half-widths;
    n_samples     : number of GP evaluations;
    problem       : the SALib "problem" dict used (bounds in normalised [0, 1]).

The companion ``plot_sobol_bar`` renders the four-bar chart requested by the
Guide (S₁ and Sₜ per parameter, with confidence intervals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SobolResult:
    names: List[str]
    S1: np.ndarray
    S1_conf: np.ndarray
    ST: np.ndarray
    ST_conf: np.ndarray
    n_samples: int
    problem: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "names": list(self.names),
            "S1": self.S1.tolist(),
            "S1_conf": self.S1_conf.tolist(),
            "ST": self.ST.tolist(),
            "ST_conf": self.ST_conf.tolist(),
            "n_samples": int(self.n_samples),
            "problem": self.problem,
        }


def _evaluate_gp_posterior_mean(model, X_np: np.ndarray) -> np.ndarray:
    """Evaluate a BoTorch model's posterior mean at numpy inputs."""
    import torch

    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.double)
        mean = model.posterior(X_t).mean.squeeze(-1).cpu().numpy()
    return mean


def compute_sobol_indices(
    model,
    *,
    active_names: Sequence[str],
    active_mask: Sequence[bool],
    full_param_order: Sequence[str],
    n_samples: int = 1024,
    calc_second_order: bool = False,
    seed: int = 0,
) -> SobolResult:
    """Compute first- and total-order Sobol indices for the GP objective.

    Parameters
    ----------
    model : botorch SingleTaskGP
        The trained objective GP from Module 3.1.
    active_names : list of str
        Parameter names that actually enter the Sobol analysis (masked-out
        dimensions are held constant at 0.5).
    active_mask : list of bool
        Positional mask against ``full_param_order``; entries False are
        pinned at 0.5.
    full_param_order : list of str
        Full parameter order the GP was trained on (PARAMETER_ORDER).
    n_samples : int
        Saltelli base sample size; the total number of GP evaluations is
        ``N · (2d + 2)`` where d is the number of active parameters.
    calc_second_order : bool
        Passed straight through to SALib.analyse.sobol.analyze.
    seed : int
        RNG seed for reproducibility.
    """
    try:
        from SALib.analyze import sobol as salib_sobol
        from SALib.sample import sobol as salib_sobol_sample
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SALib is required for Sobol indices; pip install SALib") from exc

    d_active = sum(bool(a) for a in active_mask)
    if d_active != len(active_names):
        raise ValueError("active_mask and active_names are inconsistent")
    problem = {
        "num_vars": d_active,
        "names": list(active_names),
        "bounds": [[0.0, 1.0]] * d_active,
    }

    # Generate Saltelli sample in the active subspace.
    sample_active = salib_sobol_sample.sample(
        problem, n_samples, calc_second_order=calc_second_order, seed=seed
    )

    # Embed into the full input dimension (masked-out slots pinned to 0.5).
    X_full = np.full((sample_active.shape[0], len(full_param_order)), 0.5)
    active_idx = [i for i, a in enumerate(active_mask) if a]
    X_full[:, active_idx] = sample_active

    y = _evaluate_gp_posterior_mean(model, X_full)
    Si = salib_sobol.analyze(problem, y, calc_second_order=calc_second_order, print_to_console=False)

    return SobolResult(
        names=list(active_names),
        S1=np.asarray(Si["S1"]),
        S1_conf=np.asarray(Si["S1_conf"]),
        ST=np.asarray(Si["ST"]),
        ST_conf=np.asarray(Si["ST_conf"]),
        n_samples=int(X_full.shape[0]),
        problem=problem,
    )


def plot_sobol_bar(result: SobolResult, output_path: Path, *, title: Optional[str] = None) -> Path:
    """Bar chart of S₁ and Sₜ per parameter with confidence whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result.names)
    idx = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(5.0, 0.6 * n + 2), 4.0))
    ax.bar(
        idx - width / 2, result.S1, width,
        yerr=result.S1_conf, capsize=3, label=r"$S_1$", color="tab:blue", alpha=0.8,
    )
    ax.bar(
        idx + width / 2, result.ST, width,
        yerr=result.ST_conf, capsize=3, label=r"$S_T$", color="tab:red", alpha=0.8,
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(result.names, rotation=30, ha="right")
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, max(1.0, float(np.nanmax(result.ST) + np.nanmax(result.ST_conf))) * 1.1)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.axhline(0.0, color="0.5", lw=0.5)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
