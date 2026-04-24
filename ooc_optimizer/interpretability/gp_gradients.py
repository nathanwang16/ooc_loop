"""
Module 3.3 — Local sensitivity at the optimum via GP gradients.

The Matérn-5/2 GP posterior mean has a closed-form gradient; rather than
re-deriving it, we rely on ``torch.autograd`` through the BoTorch posterior
mean.  This is exact to the GP and costs a single backward pass.

For each parameter we report ``|∂μ/∂xᵢ| · (xᵢ_max − xᵢ_min)`` — the absolute
change in predicted objective per full-range unit of that parameter.  This
normalisation makes the sensitivities directly comparable across parameters
with different physical units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LocalSensitivity:
    names: List[str]
    raw_gradient: np.ndarray           # ∂μ / ∂x_norm at the optimum (x in [0, 1])
    scaled_sensitivity: np.ndarray     # |∂μ/∂x_norm|  (== |∂μ/∂x| · range)
    ranking: List[Tuple[str, float]]   # sorted descending by scaled_sensitivity

    def to_dict(self) -> Dict:
        return {
            "names": list(self.names),
            "raw_gradient": self.raw_gradient.tolist(),
            "scaled_sensitivity": self.scaled_sensitivity.tolist(),
            "ranking": [(n, float(v)) for n, v in self.ranking],
        }


def compute_gp_gradients(
    model,
    *,
    x_optimum_norm: Sequence[float],
    active_names: Sequence[str],
    active_mask: Sequence[bool],
) -> LocalSensitivity:
    """Compute ∂μ/∂x at the normalised optimum point.

    Parameters
    ----------
    model : botorch SingleTaskGP
        The trained objective GP.
    x_optimum_norm : iterable of float
        Optimum in the same normalised [0, 1] coordinates as the GP training
        data. Its length must equal ``len(active_mask)``.
    active_names : sequence of str
        Names of the active parameters (same order as ``active_mask == True``).
    active_mask : sequence of bool
        Mask indicating which dimensions of the GP input are active.
    """
    import torch

    x0 = torch.tensor(list(x_optimum_norm), dtype=torch.double).clone().detach()
    if x0.ndim != 1 or x0.shape[0] != len(active_mask):
        raise ValueError("x_optimum_norm length must equal len(active_mask)")
    x0.requires_grad_(True)

    posterior = model.posterior(x0.unsqueeze(0))
    mu = posterior.mean.squeeze()
    (grad,) = torch.autograd.grad(mu, x0, create_graph=False, retain_graph=False)
    grad_np = grad.detach().cpu().numpy()

    active_idx = [i for i, a in enumerate(active_mask) if a]
    if len(active_idx) != len(active_names):
        raise ValueError("active_mask / active_names length mismatch")
    raw = grad_np[active_idx]

    # Because the GP operates in normalised [0, 1] coordinates, the raw
    # gradient already represents "change in objective per full range of
    # parameter"; no extra scaling needed.
    scaled = np.abs(raw)
    order = np.argsort(-scaled)
    ranking = [(active_names[i], float(scaled[i])) for i in order]
    return LocalSensitivity(
        names=list(active_names),
        raw_gradient=raw,
        scaled_sensitivity=scaled,
        ranking=ranking,
    )


def plot_local_sensitivity(result: LocalSensitivity, output_path: Path, *, title: Optional[str] = None) -> Path:
    """Horizontal bar chart ranking parameters by |∂μ/∂x_norm|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in result.ranking]
    vals = [v for _, v in result.ranking]

    fig, ax = plt.subplots(figsize=(6.0, max(2.5, 0.4 * len(names) + 1)))
    ax.barh(range(len(names))[::-1], vals, color="tab:purple", alpha=0.8)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$|\partial\mu/\partial x_{\mathrm{norm}}|$ at optimum")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
