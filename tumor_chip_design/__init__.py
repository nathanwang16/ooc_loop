"""
Public namespace for the tumor-chip-design package.

This module re-exports the underlying implementation package
``ooc_optimizer`` under the manuscript-facing name so users who follow the
publication can ``from tumor_chip_design import ...`` directly.  Both
names point at the same objects; there is no functional difference.

A future release (v0.7.0, tracked in CHANGELOG.md) will migrate
``ooc_optimizer`` onto the full ``src/tumor_chip_design/`` src-layout
described in Development Guide v2 §5.1.  Until then, this compatibility
shim keeps the public API stable for downstream users while avoiding a
disruptive cross-repo rename.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

__version__ = "0.6.0"

_IMPL_PACKAGE = "ooc_optimizer"
_SUBMODULES = (
    "cfd",
    "config",
    "geometry",
    "optimization",
    "analysis",
    "interpretability",
    "validation",
)

# Re-export top-level package.  Accessing tumor_chip_design.cfd triggers the
# normal import machinery on ooc_optimizer.cfd because of the alias below.
_impl = importlib.import_module(_IMPL_PACKAGE)
for _name in _SUBMODULES:
    _sub = importlib.import_module(f"{_IMPL_PACKAGE}.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _sub
    globals()[_name] = _sub


def __getattr__(name: str) -> Any:
    """Fall-through attribute lookup that mirrors ``ooc_optimizer``."""
    if hasattr(_impl, name):
        return getattr(_impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ("__version__", *_SUBMODULES)
