"""
Module 2.2 / 3.1 — Target-profile definitions and loss functions.

The v2 BO objective is the relative L2 distance between the CFD-produced
concentration field C(x, y) on the chamber floor and a user-specified target
profile C_target(x, y).  Target profiles live here so the same definitions
are used by the orchestrator (Module 2.2), the BO loop (Module 3.1), the
interpretability analysis (Module 3.3), and the plotting code (Module 3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

TargetKind = Literal["linear_gradient", "bimodal", "step", "custom"]


@dataclass
class TargetProfile:
    """A target concentration profile evaluated on arbitrary (x, y) samples.

    The callable signature is ``C_target(x, y, *, L, W) -> ndarray`` where
    ``L`` and ``W`` are the chamber length and width in metres.  Both ``x``
    and ``y`` are arrays in metres.
    """

    kind: TargetKind
    params: Dict[str, float]
    _fn: Callable[..., np.ndarray]

    def evaluate(self, x: np.ndarray, y: np.ndarray, *, L: float, W: float) -> np.ndarray:
        return self._fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float), L=L, W=W, **self.params)


def linear_gradient(
    axis: str = "x",
    c_high: float = 1.0,
    c_low: float = 0.0,
) -> TargetProfile:
    """Monotonic linear gradient along `axis` (``"x"`` or ``"y"``).

    C_target(ξ) = c_low + (c_high − c_low) · ξ,  ξ = x/L or y/W.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

    def _eval(x, y, *, L, W, axis, c_high, c_low):  # noqa: ARG001
        xi = (x / L) if axis == "x" else (y / W)
        return c_low + (c_high - c_low) * xi

    return TargetProfile(
        kind="linear_gradient",
        params={"axis": axis, "c_high": c_high, "c_low": c_low},
        _fn=_eval,
    )


def bimodal(
    peak_axis: str = "x",
    peak_fracs: Tuple[float, float] = (0.25, 0.75),
    width_frac: float = 0.1,
    c_peak: float = 1.0,
    c_base: float = 0.0,
) -> TargetProfile:
    """Sum of two Gaussian bumps along `peak_axis`.

    C_target(ξ) = c_base + (c_peak − c_base) · max(G₁(ξ), G₂(ξ))
        where Gᵢ(ξ) = exp(−((ξ − μᵢ) / σ)²), μᵢ = peak_fracs[i], σ = width_frac.
    """
    if peak_axis not in ("x", "y"):
        raise ValueError(f"peak_axis must be 'x' or 'y', got '{peak_axis}'")
    if width_frac <= 0:
        raise ValueError("width_frac must be positive")
    p0, p1 = peak_fracs
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError("peak fractions must be in (0, 1)")

    def _eval(x, y, *, L, W, peak_axis, peak_fracs, width_frac, c_peak, c_base):  # noqa: ARG001
        xi = (x / L) if peak_axis == "x" else (y / W)
        g1 = np.exp(-((xi - peak_fracs[0]) / width_frac) ** 2)
        g2 = np.exp(-((xi - peak_fracs[1]) / width_frac) ** 2)
        return c_base + (c_peak - c_base) * np.maximum(g1, g2)

    return TargetProfile(
        kind="bimodal",
        params={
            "peak_axis": peak_axis,
            "peak_fracs": peak_fracs,
            "width_frac": width_frac,
            "c_peak": c_peak,
            "c_base": c_base,
        },
        _fn=_eval,
    )


def step(
    step_axis: str = "x",
    step_frac: float = 0.5,
    sharpness_frac: float = 0.01,
    c_high: float = 1.0,
    c_low: float = 0.0,
) -> TargetProfile:
    """Sigmoid step transition at ``step_frac``.

    C_target(ξ) = c_low + (c_high − c_low) · σ((ξ − step_frac) / sharpness_frac)
    where σ is the logistic function.
    """
    if step_axis not in ("x", "y"):
        raise ValueError(f"step_axis must be 'x' or 'y', got '{step_axis}'")
    if not 0.0 < step_frac < 1.0:
        raise ValueError("step_frac must be in (0, 1)")
    if sharpness_frac <= 0:
        raise ValueError("sharpness_frac must be positive")

    def _eval(x, y, *, L, W, step_axis, step_frac, sharpness_frac, c_high, c_low):  # noqa: ARG001
        xi = (x / L) if step_axis == "x" else (y / W)
        sig = 1.0 / (1.0 + np.exp(-(xi - step_frac) / sharpness_frac))
        return c_low + (c_high - c_low) * sig

    return TargetProfile(
        kind="step",
        params={
            "step_axis": step_axis,
            "step_frac": step_frac,
            "sharpness_frac": sharpness_frac,
            "c_high": c_high,
            "c_low": c_low,
        },
        _fn=_eval,
    )


def custom(fn: Callable[..., np.ndarray], name: str = "custom") -> TargetProfile:
    """Wrap a user-supplied ``fn(x, y, *, L, W) -> ndarray``."""

    def _eval(x, y, *, L, W):
        return fn(x, y, L=L, W=W)

    return TargetProfile(kind="custom", params={"name": name}, _fn=_eval)


# ---------------------------------------------------------------------------
# Factory dispatching from a pydantic/yaml-like config.
# ---------------------------------------------------------------------------


def build_target_profile(spec: Dict[str, object]) -> TargetProfile:
    """Build a TargetProfile from a dict spec (as stored in the YAML config).

    The dict must have a ``kind`` key; remaining keys are passed to the
    matching constructor.
    """
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError("target profile spec requires a 'kind' key")
    if kind == "linear_gradient":
        return linear_gradient(**spec)
    if kind == "bimodal":
        # peak_fracs needs tuple coercion when loaded from yaml.
        if "peak_fracs" in spec:
            spec["peak_fracs"] = tuple(spec["peak_fracs"])
        return bimodal(**spec)
    if kind == "step":
        return step(**spec)
    raise ValueError(f"Unknown target profile kind '{kind}' (use custom(...) in Python)")


# ---------------------------------------------------------------------------
# Loss functions and diagnostics on the achieved concentration field.
# ---------------------------------------------------------------------------


def l2_to_target(
    C_sim: np.ndarray,
    C_target: np.ndarray,
    *,
    cell_weights: Optional[np.ndarray] = None,
) -> float:
    """Relative L2 distance between simulated and target fields.

        L2_rel = sqrt( Σ wᵢ (Cᵢ − Cᵢ_target)² / Σ wᵢ )
                 -------------------------------------
                        sqrt( Σ wᵢ Cᵢ_target² / Σ wᵢ ) + ε

    When `cell_weights` is None, equal weighting is used.
    """
    C_sim = np.asarray(C_sim, dtype=float)
    C_target = np.asarray(C_target, dtype=float)
    if C_sim.shape != C_target.shape:
        raise ValueError(f"Field shape mismatch: {C_sim.shape} vs {C_target.shape}")
    w = cell_weights if cell_weights is not None else np.ones_like(C_sim)
    w_total = float(np.sum(w))
    if w_total <= 0:
        raise ValueError("cell_weights must sum to a positive value")
    diff_rms = float(np.sqrt(np.sum(w * (C_sim - C_target) ** 2) / w_total))
    tgt_rms = float(np.sqrt(np.sum(w * C_target ** 2) / w_total))
    return diff_rms / (tgt_rms + 1e-12)


def gradient_sharpness(
    C: np.ndarray,
    centres: np.ndarray,
    *,
    L: float,
) -> float:
    """Mean ‖∇C‖ estimate, normalised by 1/L.

    Implementation: uses nearest-neighbour finite differences via a regular-
    grid projection — we sort centres in x and estimate dC/dx with numpy
    gradient.  For irregular meshes this is a coarse diagnostic (intentional;
    the metric is only used as a heuristic, not a constraint).
    """
    C = np.asarray(C, dtype=float)
    x = centres[:, 0]
    order = np.argsort(x)
    x_s = x[order]
    C_s = C[order]
    # drop duplicate x values
    uniq_mask = np.concatenate(([True], np.diff(x_s) > 1e-9))
    if uniq_mask.sum() < 2:
        return 0.0
    dCdx = np.gradient(C_s[uniq_mask], x_s[uniq_mask])
    return float(np.mean(np.abs(dCdx)) * L)


def monotonicity_fraction(
    C: np.ndarray,
    centres: np.ndarray,
    *,
    axis: str = "x",
) -> float:
    """Fraction of adjacent cell-pairs whose ∂C/∂axis has a consistent sign.

    Parameters
    ----------
    axis : {"x", "y"}
        Axis along which to measure monotonicity.

    Returns
    -------
    frac : float
        Maximum of (fraction with positive slope, fraction with negative
        slope).  1.0 indicates perfect monotonicity; 0.5 indicates random.
    """
    if axis == "x":
        coord = centres[:, 0]
    elif axis == "y":
        coord = centres[:, 1]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
    order = np.argsort(coord)
    C_s = np.asarray(C, dtype=float)[order]
    diffs = np.diff(C_s)
    if diffs.size == 0:
        return 0.0
    frac_pos = float(np.mean(diffs >= 0))
    frac_neg = float(np.mean(diffs <= 0))
    return max(frac_pos, frac_neg)
