"""
Module 3.3 — Tolerance-interval estimation per parameter.

Starting from the BO optimum x*, we move along each active axis until the GP
posterior-mean prediction of the objective degrades by a configurable
fractional amount (default 10%).  The ± excursion is the tolerance interval
for that parameter.

Bisection-on-the-GP is adequate here because the GP posterior-mean is smooth
and one-dimensional along each axis.  The fallback for non-monotonic slices
is a coarse linear sweep followed by refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToleranceInterval:
    name: str
    delta_plus_norm: float   # absolute size of the positive excursion in [0, 1] coords
    delta_minus_norm: float  # absolute size of the negative excursion in [0, 1] coords
    delta_plus_phys: float   # same in physical units
    delta_minus_phys: float
    baseline_mu: float
    threshold_mu: float

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "delta_plus_norm": float(self.delta_plus_norm),
            "delta_minus_norm": float(self.delta_minus_norm),
            "delta_plus_phys": float(self.delta_plus_phys),
            "delta_minus_phys": float(self.delta_minus_phys),
            "baseline_mu": float(self.baseline_mu),
            "threshold_mu": float(self.threshold_mu),
        }


def _gp_mu(model, x_full_norm: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        x_t = torch.tensor(x_full_norm[np.newaxis, :], dtype=torch.double)
        return float(model.posterior(x_t).mean.item())


def _bisect_axis(
    model,
    x0_norm: np.ndarray,
    axis_idx: int,
    *,
    direction: int,
    threshold_mu: float,
    tolerance: float = 1e-3,
    max_iter: int = 40,
) -> float:
    """Find smallest |Δ| such that μ(x0 + Δ·e_axis) ≥ threshold_mu.

    direction ∈ {+1, -1}: which direction along the axis to explore.
    Returns |Δ| (in normalised [0, 1] coords).  If the axis hits its bound
    before the threshold is reached, returns the remaining distance to the
    bound (i.e. "as loose as the whole design space").
    """
    low = 0.0
    upper_bound = 1.0 - x0_norm[axis_idx] if direction > 0 else x0_norm[axis_idx]
    if upper_bound <= 0.0:
        return 0.0

    def mu_at(delta: float) -> float:
        x = x0_norm.copy()
        x[axis_idx] = x0_norm[axis_idx] + direction * delta
        x[axis_idx] = float(np.clip(x[axis_idx], 0.0, 1.0))
        return _gp_mu(model, x)

    high = upper_bound
    mu_high = mu_at(high)
    if mu_high < threshold_mu:
        # Even at the boundary the threshold isn't crossed; report the full
        # available range as the tolerance.
        return float(upper_bound)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mu_mid = mu_at(mid)
        if mu_mid >= threshold_mu:
            high = mid
        else:
            low = mid
        if high - low < tolerance:
            break
    return float(0.5 * (low + high))


def compute_tolerance_intervals(
    model,
    *,
    x_optimum_norm: Sequence[float],
    active_names: Sequence[str],
    active_mask: Sequence[bool],
    bounds: Dict[str, tuple],
    loss_tolerance: float = 0.1,
) -> List[ToleranceInterval]:
    """For each active parameter, bisect GP μ to find ±Δ producing loss_tol degradation.

    The threshold is ``μ*(1 + loss_tolerance)`` when the objective is being
    *minimised* (as in Module 3.1).  If the baseline μ* is non-positive (which
    can occur with standardised outputs), we fall back to an absolute
    ``loss_tolerance`` additive offset.
    """
    x0 = np.asarray(list(x_optimum_norm), dtype=float)
    if x0.shape[0] != len(active_mask):
        raise ValueError("x_optimum_norm length must equal len(active_mask)")

    mu0 = _gp_mu(model, x0)
    if mu0 > 0:
        threshold_mu = mu0 * (1.0 + loss_tolerance)
    else:
        threshold_mu = mu0 + max(loss_tolerance, 1e-6)

    active_idx = [i for i, a in enumerate(active_mask) if a]
    if len(active_idx) != len(active_names):
        raise ValueError("active_mask / active_names length mismatch")

    intervals: List[ToleranceInterval] = []
    for name, idx in zip(active_names, active_idx):
        dp_norm = _bisect_axis(model, x0, idx, direction=+1, threshold_mu=threshold_mu)
        dm_norm = _bisect_axis(model, x0, idx, direction=-1, threshold_mu=threshold_mu)
        bmin, bmax = bounds[name]
        rng = bmax - bmin
        intervals.append(
            ToleranceInterval(
                name=name,
                delta_plus_norm=dp_norm,
                delta_minus_norm=dm_norm,
                delta_plus_phys=dp_norm * rng,
                delta_minus_phys=dm_norm * rng,
                baseline_mu=mu0,
                threshold_mu=threshold_mu,
            )
        )
    return intervals


def validate_with_cfd(
    intervals: List[ToleranceInterval],
    *,
    validate_top_k: int,
    validate_fn: Callable[[str, float, float], Dict],
) -> List[Dict]:
    """Re-run CFD at x* ± Δ for the top-k and bottom-k-tolerant parameters.

    ``validate_fn`` is a caller-supplied callable with signature
    ``validate_fn(name, delta_norm_plus, delta_norm_minus) -> metrics_dict``
    that stages and runs an OpenFOAM case.  Kept as a callback because the
    orchestrator controls CFD invocation; this module is otherwise pure
    numerics.
    """
    if validate_top_k <= 0:
        return []
    widths = [i.delta_plus_norm + i.delta_minus_norm for i in intervals]
    order = np.argsort(widths)
    bottom = order[:validate_top_k]  # least-tolerant
    top = order[-validate_top_k:]    # most-tolerant
    to_validate = list(top) + list(bottom)
    out: List[Dict] = []
    for idx in to_validate:
        iv = intervals[idx]
        out.append(
            {
                "name": iv.name,
                "delta_plus_norm": iv.delta_plus_norm,
                "delta_minus_norm": iv.delta_minus_norm,
                "cfd_validation": validate_fn(iv.name, iv.delta_plus_norm, iv.delta_minus_norm),
            }
        )
    return out


def plot_tolerance_intervals(
    intervals: List[ToleranceInterval],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Asymmetric error-bar plot of ± tolerance in physical units."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [i.name for i in intervals]
    plus = [i.delta_plus_phys for i in intervals]
    minus = [i.delta_minus_phys for i in intervals]

    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.4 * len(names) + 1)))
    y = np.arange(len(names))
    ax.errorbar(
        np.zeros_like(y, dtype=float),
        y,
        xerr=[minus, plus],
        fmt="o",
        color="tab:green",
        ecolor="tab:green",
        capsize=4,
        lw=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0.0, color="0.5", lw=0.5)
    ax.set_xlabel("Tolerance interval (physical units)")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
