"""CFD package public API."""

from __future__ import annotations

from typing import Any

__all__ = [
    "evaluate_cfd",
    "run_scalar_transport",
    "run_scalar_verification_1d",
    "analytic_ad_1d",
    "extract_concentration_field",
]


def evaluate_cfd(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy-import solver to avoid geometry dependency on package import."""
    from ooc_optimizer.cfd.solver import evaluate_cfd as _evaluate_cfd

    return _evaluate_cfd(*args, **kwargs)


def run_scalar_transport(*args: Any, **kwargs: Any):
    from ooc_optimizer.cfd.scalar import run_scalar_transport as _fn

    return _fn(*args, **kwargs)


def run_scalar_verification_1d(*args: Any, **kwargs: Any):
    from ooc_optimizer.cfd.scalar import run_scalar_verification_1d as _fn

    return _fn(*args, **kwargs)


def analytic_ad_1d(*args: Any, **kwargs: Any):
    from ooc_optimizer.cfd.scalar import analytic_ad_1d as _fn

    return _fn(*args, **kwargs)


def extract_concentration_field(*args: Any, **kwargs: Any):
    from ooc_optimizer.cfd.scalar import extract_concentration_field as _fn

    return _fn(*args, **kwargs)
