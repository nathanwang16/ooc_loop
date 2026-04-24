"""Module 5.2 — tumor_chip_design compatibility namespace shim."""

from __future__ import annotations

import pytest


def test_public_namespace_imports():
    import tumor_chip_design  # noqa: F401

    assert tumor_chip_design.__version__


def test_submodules_are_aliased():
    import tumor_chip_design

    for name in ("cfd", "config", "geometry", "optimization", "analysis",
                "interpretability", "validation"):
        sub = getattr(tumor_chip_design, name)
        assert sub.__name__.startswith("ooc_optimizer."), (
            f"{name} must alias ooc_optimizer.{name} (got {sub.__name__})"
        )


def test_reexported_builder_matches_impl():
    import tumor_chip_design as tcd
    from ooc_optimizer.geometry.topology_blockmesh import (
        generate_blockmesh_dict_v2 as impl,
    )

    assert tcd.geometry.generate_blockmesh_dict_v2 is impl


def test_missing_attribute_raises_attribute_error():
    import tumor_chip_design as tcd

    with pytest.raises(AttributeError):
        tcd.this_attribute_does_not_exist  # type: ignore[attr-defined]
