"""
Module 2.3 — Configuration schema (v2).

Pydantic models validate the YAML config at load time rather than at
mid-run, and carry through to IDE autocomplete / automatic docs
generation.  The public ``load_config`` function returns both the raw
dictionary (for unchanged legacy callers) and, on request, the parsed
pydantic model.

Schema scope (v2):
    - continuous bounds for 7 parameters (W, d_p, s_p, theta, Q_total,
      r_flow, delta_W), with per-topology masks on which are active;
    - discrete levels across pillar_config × chamber_height × inlet_topology;
    - solver settings (unchanged from v1) + passive-scalar diffusivity;
    - target-profile spec (linear_gradient / bimodal / step);
    - optimization block (Sobol/BO counts, constraint bounds, penalty);
    - paths block (unchanged from v1, but validated);
    - interpretability sub-block (Module 3.3).

The schema is intentionally permissive for backward compatibility with v1
configs: ``Q`` is accepted as an alias for ``Q_total``, ``opposing`` is the
default topology, and an absent target profile defaults to a linear
x-gradient from 1 to 0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------


class ParamBounds(BaseModel):
    """Minimum / maximum for one continuous optimizer input."""

    min: float
    max: float

    @model_validator(mode="after")
    def _min_lt_max(self):
        if self.min >= self.max:
            raise ValueError(f"min >= max: min ({self.min}) must be < max ({self.max})")
        return self


class ContinuousBounds(BaseModel):
    model_config = ConfigDict(extra="allow")

    W: ParamBounds
    d_p: ParamBounds
    s_p: ParamBounds
    theta: ParamBounds
    Q_total: ParamBounds
    r_flow: ParamBounds
    delta_W: ParamBounds


class DiscreteLevels(BaseModel):
    pillar_config: List[Literal["none", "1x4", "2x4", "3x6"]]
    chamber_height: List[float]
    inlet_topology: List[Literal["opposing", "same_side_Y", "asymmetric_lumen"]]

    @field_validator("chamber_height")
    @classmethod
    def _heights_positive(cls, v):
        if not v or any(h <= 0 for h in v):
            raise ValueError("chamber_height must be a non-empty list of positive values")
        return v


# ---------------------------------------------------------------------------
# Target profiles
# ---------------------------------------------------------------------------


class LinearGradientSpec(BaseModel):
    kind: Literal["linear_gradient"] = "linear_gradient"
    axis: Literal["x", "y"] = "x"
    c_high: float = 1.0
    c_low: float = 0.0


class BimodalSpec(BaseModel):
    kind: Literal["bimodal"] = "bimodal"
    peak_axis: Literal["x", "y"] = "x"
    peak_fracs: Tuple[float, float] = (0.25, 0.75)
    width_frac: float = 0.1
    c_peak: float = 1.0
    c_base: float = 0.0


class StepSpec(BaseModel):
    kind: Literal["step"] = "step"
    step_axis: Literal["x", "y"] = "x"
    step_frac: float = 0.5
    sharpness_frac: float = 0.01
    c_high: float = 1.0
    c_low: float = 0.0


TargetProfileSpec = Union[LinearGradientSpec, BimodalSpec, StepSpec]


# ---------------------------------------------------------------------------
# Solver / fixed / opt / interp blocks
# ---------------------------------------------------------------------------


class FixedParameters(BaseModel):
    model_config = ConfigDict(extra="allow")
    chamber_length_um: float = 10000.0
    inlet_width_um: float = 500.0
    fluid_viscosity_Pa_s: float = 1e-3
    fluid_density_kg_m3: float = 1000.0


class SolverSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    convergence_criterion: float = 1e-6
    max_iterations: int = 2000
    mesh_resolution: int = 1
    solver_timeout_s: int = 300


class ConstraintBounds(BaseModel):
    tau_mean_min: float = 0.1
    tau_mean_max: float = 2.0
    f_dead_max: float = 0.05


class OptimizationBlock(BaseModel):
    n_sobol_init: int = 20
    n_bo_iterations: int = 40
    total_per_config: int = 60
    constraints: ConstraintBounds = Field(default_factory=ConstraintBounds)
    penalty_L2: float = 99.0
    penalty_cv_tau: float = 999.0  # retained for v1 compatibility


class InterpretabilityConfig(BaseModel):
    sobol_n_samples: int = 1024
    tolerance_loss_tolerance: float = 0.1  # fractional degradation of L2
    validate_top_k: int = 3
    bootstrap_n: int = 200


class PathsBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    template_case: str
    stl_output_dir: str = "data/stl"
    case_output_dir: str = "data/cases"
    results_dir: str = "data/results"
    evaluation_log: str = "data/results/evaluations.jsonl"
    figures_dir: str = "figures"


class Baseline(BaseModel):
    model_config = ConfigDict(extra="allow")
    W: float
    theta: float
    Q_total: float
    r_flow: float = 0.5
    delta_W: float = 0.2
    d_p: float = 0.0
    s_p: float = 0.0
    pillar_config: str = "none"
    H: float = 200.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class OocConfigV2(BaseModel):
    model_config = ConfigDict(extra="allow")

    fixed_parameters: FixedParameters
    continuous_bounds: ContinuousBounds
    discrete_levels: DiscreteLevels
    solver_settings: SolverSettings
    optimization: OptimizationBlock = Field(default_factory=OptimizationBlock)
    paths: PathsBlock
    baseline: Optional[Baseline] = None

    diffusivity: float = 1e-10
    target_profile: TargetProfileSpec = Field(default_factory=LinearGradientSpec)
    interpretability: InterpretabilityConfig = Field(default_factory=InterpretabilityConfig)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _normalize_legacy_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a v1 dict so that pydantic v2 validation succeeds.

    Accepts a v1 config lacking r_flow / delta_W / inlet_topology /
    target_profile and fills in v2 defaults in-place.
    """
    cfg = dict(cfg)
    cont = dict(cfg.get("continuous_bounds", {}))
    if "Q" in cont and "Q_total" not in cont:
        cont["Q_total"] = cont.pop("Q")
    cont.setdefault("r_flow", {"min": 0.1, "max": 0.9})
    cont.setdefault("delta_W", {"min": 0.1, "max": 0.45})
    cfg["continuous_bounds"] = cont

    disc = dict(cfg.get("discrete_levels", {}))
    disc.setdefault("inlet_topology", ["opposing", "same_side_Y", "asymmetric_lumen"])
    cfg["discrete_levels"] = disc

    opt = dict(cfg.get("optimization", {}))
    if "n_sobol_init" in opt and opt["n_sobol_init"] < 20:
        # v2 requires at least 20 Sobol points for the larger design space.
        opt.setdefault("n_sobol_init", 20)
    cfg["optimization"] = opt
    return cfg


def load_config(config_path: Path, *, as_model: bool = False):
    """Load a YAML config file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML file.
    as_model : bool
        When True, return the validated ``OocConfigV2`` instance rather than
        the raw dict.  Defaults to False for backward compatibility with the
        v1 codepath.

    Returns
    -------
    config : dict or OocConfigV2
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError("Configuration file is empty")

    raw = _normalize_legacy_fields(raw)
    try:
        model = OocConfigV2.model_validate(raw)
    except Exception as exc:  # re-raise as ValueError with a clearer prefix
        msg = str(exc)
        # Remap pydantic's phrasing to the v1 "Missing required..." prefix the
        # legacy tests and CLI drivers still check for.
        if "Field required" in msg:
            msg = "Missing required field(s); details:\n" + msg
        raise ValueError(msg) from exc
    logger.info("Configuration loaded from %s (topologies=%s)", config_path, model.discrete_levels.inlet_topology)

    if as_model:
        return model
    return _model_to_dict(model, raw)


def _model_to_dict(model: OocConfigV2, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a validated model back to a dict while preserving raw extras."""
    dumped = model.model_dump()
    # Preserve raw keys the schema doesn't know about (e.g. user-defined
    # extensions); pydantic v2 with extra="allow" already does this but the
    # dump round-trip drops some snappy_* flags that live under solver_settings.
    for block in ("fixed_parameters", "solver_settings"):
        if block in raw and isinstance(raw[block], dict):
            dumped[block] = {**dumped.get(block, {}), **raw[block]}
    # target_profile is typed via discriminated union; simplify to a plain dict.
    tp = dumped.get("target_profile")
    if isinstance(tp, dict) and "kind" not in tp:
        # Pydantic sometimes writes the discriminator under its tag.
        tp = {**tp}
    return dumped
