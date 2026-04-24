"""
Module 3.1 — Bayesian optimization loop (v2).

One BORunner == one (pillar_config, chamber_height, inlet_topology) triple.
Objective: L2_to_target (minimised).
Constraints (encoded as slack ≥ 0):
    c1 = tau_mean - tau_mean_min     (flow not pathologically slow)
    c2 = tau_mean_max - tau_mean     (flow not pathologically fast)
    c3 = f_dead_max - f_dead         (no dead zones)

The continuous parameter vector is topology-aware: for ``opposing`` we
optimise over 7 parameters (W, d_p, s_p, theta, Q_total, r_flow, delta_W);
for the other two topologies delta_W is masked out (6 parameters).  When a
pillar_config is ``"none"`` the pillar-only parameters (d_p, s_p) are also
masked out; they still appear in the BO vector but are pinned to their
midpoint so the GP does not receive informationless inputs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ooc_optimizer.config.logger import EvaluationLogger
from ooc_optimizer.optimization.objectives import TargetProfile, build_target_profile

logger = logging.getLogger(__name__)

N_SOBOL_INIT = 20
N_BO_ITERATIONS = 40
N_ACQ_SAMPLES = 2048

# Master parameter order used for all topologies.  Entries masked out for a
# given topology are still present in the BO vector (pinned to midpoint)
# so that per-topology GPs share a common input dimensionality for plotting
# and cross-topology analysis.
PARAMETER_ORDER = ["W", "d_p", "s_p", "theta", "Q_total", "r_flow", "delta_W"]


def _active_params(pillar_config: str, topology: str) -> List[bool]:
    """Boolean mask saying which PARAMETER_ORDER slots are active."""
    active = [True] * len(PARAMETER_ORDER)
    if pillar_config.lower() == "none":
        active[PARAMETER_ORDER.index("d_p")] = False
        active[PARAMETER_ORDER.index("s_p")] = False
    if topology != "opposing":
        active[PARAMETER_ORDER.index("delta_W")] = False
    return active


class BORunner:
    """Bayesian optimization loop for one (pillar_config, H, topology) combination."""

    def __init__(
        self,
        config: dict,
        pillar_config: str,
        H: float,
        *,
        topology: str = "opposing",
        target_profile: Optional[TargetProfile] = None,
    ):
        if config is None:
            raise ValueError("config must not be None")
        if pillar_config not in {"none", "1x4", "2x4", "3x6"}:
            raise ValueError(f"Invalid pillar_config: {pillar_config}")
        if H not in (200.0, 300.0):
            raise ValueError(f"Invalid chamber height: {H}")
        if topology not in {"opposing", "same_side_Y", "asymmetric_lumen"}:
            raise ValueError(f"Invalid topology: {topology}")
        for key in ("continuous_bounds", "optimization", "paths"):
            if key not in config:
                raise ValueError(f"Missing '{key}' in config")

        self.config = config
        self.pillar_config = pillar_config
        self.H = H
        self.topology = topology

        self._bounds = self._extract_bounds(config["continuous_bounds"])
        self._constraint_config = self._extract_constraint_config(config["optimization"])
        self.n_sobol = int(config["optimization"].get("n_sobol_init", N_SOBOL_INIT))
        self.n_bo = int(config["optimization"].get("n_bo_iterations", N_BO_ITERATIONS))
        self.penalty_L2 = float(config["optimization"].get("penalty_L2", 99.0))
        self.results_dir = Path(config["paths"]["results_dir"])

        self._cfd_config = self._build_cfd_config(config)
        self.logger = self._build_evaluation_logger(config)

        if target_profile is None:
            target_profile = build_target_profile(dict(config.get("target_profile", {"kind": "linear_gradient"})))
        self.target_profile = target_profile

        self._active_mask = _active_params(pillar_config, topology)
        self._n_active = sum(self._active_mask)

        self.train_X = None
        self.train_Y = None
        self.train_constraints = None
        self.evaluations: List[Dict[str, Any]] = []
        self._objective_model = None
        self._constraint_models: List[Any] = []
        self._bo_deps: Optional[Dict[str, Any]] = None

    # --------------------------- dependency handling -----------------------

    def _require_bo_stack(self) -> Dict[str, Any]:
        if self._bo_deps is not None:
            return self._bo_deps
        try:
            import torch
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from torch.distributions import Normal
            from torch.quasirandom import SobolEngine
        except ImportError as exc:
            raise ImportError(
                "BO dependencies missing. Install requirements.txt (torch, gpytorch, botorch)."
            ) from exc
        self._bo_deps = {
            "torch": torch,
            "fit_gpytorch_mll": fit_gpytorch_mll,
            "SingleTaskGP": SingleTaskGP,
            "Standardize": Standardize,
            "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
            "Normal": Normal,
            "SobolEngine": SobolEngine,
        }
        return self._bo_deps

    # --------------------------- config extractors -------------------------

    @staticmethod
    def _extract_bounds(bounds_cfg: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for name in PARAMETER_ORDER:
            if name not in bounds_cfg:
                raise ValueError(f"Missing continuous bounds for '{name}'")
            entry = bounds_cfg[name]
            if "min" not in entry or "max" not in entry:
                raise ValueError(f"Bounds for '{name}' must include min and max")
            bmin = float(entry["min"])
            bmax = float(entry["max"])
            if bmin >= bmax:
                raise ValueError(f"Invalid bounds for '{name}': min >= max")
            bounds[name] = (bmin, bmax)
        return bounds

    @staticmethod
    def _extract_constraint_config(opt_cfg: Dict[str, Any]) -> Dict[str, float]:
        if "constraints" not in opt_cfg:
            raise ValueError("Missing optimization.constraints in config")
        c = opt_cfg["constraints"]
        for key in ("tau_mean_min", "tau_mean_max", "f_dead_max"):
            if key not in c:
                raise ValueError(f"Missing constraint key '{key}'")
        return {k: float(c[k]) for k in ("tau_mean_min", "tau_mean_max", "f_dead_max")}

    @staticmethod
    def _build_cfd_config(config: Dict[str, Any]) -> Dict[str, Any]:
        paths = config["paths"]
        for key in ("case_output_dir", "template_case"):
            if key not in paths:
                raise ValueError(f"Missing path key '{key}' required for CFD")
        cfd_config = dict(config)
        cfd_paths = dict(paths)
        cfd_paths["work_dir"] = paths["case_output_dir"]
        cfd_paths["template_dir"] = paths["template_case"]
        cfd_config["paths"] = cfd_paths
        fixed = config.get("fixed_parameters")
        if fixed is None or "fluid_viscosity_Pa_s" not in fixed:
            raise ValueError("Missing fixed_parameters.fluid_viscosity_Pa_s")
        cfd_config["diffusivity"] = float(config.get("diffusivity", 1e-10))
        return cfd_config

    def _build_evaluation_logger(self, config: Dict[str, Any]) -> EvaluationLogger:
        eval_log_path = Path(config["paths"]["evaluation_log"])
        stem = eval_log_path.stem
        suffix = eval_log_path.suffix or ".jsonl"
        scoped_name = f"{stem}_{self.topology}_{self.pillar_config}_H{int(self.H)}{suffix}"
        scoped_log = eval_log_path.with_name(scoped_name)
        return EvaluationLogger(scoped_log)

    # --------------------------- main loop ---------------------------------

    def run(self) -> Dict:
        """Execute Sobol init + BO iterations; return full results dict."""
        logger.info(
            "Starting BO: topology=%s, pillar=%s, H=%.0f (Sobol=%d, BO=%d, active_dim=%d)",
            self.topology, self.pillar_config, self.H, self.n_sobol, self.n_bo, self._n_active,
        )
        deps = self._require_bo_stack()
        torch = deps["torch"]

        init_points = self._generate_sobol_points(self.n_sobol)
        X_rows: List[Any] = []
        Y_rows: List[float] = []
        C_rows: List[List[float]] = []
        for x in init_points:
            objective, constraints = self._evaluate_point(x)
            X_rows.append(x)
            Y_rows.append(objective)
            C_rows.append(constraints)

        self.train_X = torch.stack(X_rows).to(dtype=torch.double)
        self.train_Y = torch.tensor(Y_rows, dtype=torch.double).unsqueeze(-1)
        self.train_constraints = torch.tensor(C_rows, dtype=torch.double)

        for iteration in range(self.n_bo):
            self._fit_gp()
            x_next = self._optimize_acquisition()
            objective, constraints = self._evaluate_point(x_next)

            self.train_X = torch.cat([self.train_X, x_next.unsqueeze(0)], dim=0)
            self.train_Y = torch.cat(
                [self.train_Y, torch.tensor([[objective]], dtype=torch.double)], dim=0
            )
            self.train_constraints = torch.cat(
                [self.train_constraints, torch.tensor([constraints], dtype=torch.double)], dim=0
            )
            logger.info(
                "BO iter %d/%d (%s, %s, H=%.0f): L2=%.4f",
                iteration + 1, self.n_bo, self.topology, self.pillar_config, self.H, objective,
            )

        best = self._get_best_feasible()
        state_dir = self.results_dir / f"bo_{self.topology}_{self.pillar_config}_H{int(self.H)}"
        self.save_state(state_dir)

        return {
            "config_name": f"{self.topology}_{self.pillar_config}_H{int(self.H)}",
            "topology": self.topology,
            "pillar_config": self.pillar_config,
            "H": self.H,
            "target_profile": {"kind": self.target_profile.kind, **self.target_profile.params},
            "n_evaluations": len(self.evaluations),
            "best_feasible": best,
            "best_L2": best["L2_to_target"] if best else None,
            "best_params": best["params"] if best else None,
            "state_dir": str(state_dir),
            "evaluations": self.evaluations,
        }

    # --------------------------- helpers -----------------------------------

    def _generate_sobol_points(self, n: int):
        if n <= 0:
            raise ValueError("n must be > 0 for Sobol generation")
        deps = self._require_bo_stack()
        SobolEngine = deps["SobolEngine"]
        torch = deps["torch"]
        sobol = SobolEngine(dimension=len(PARAMETER_ORDER), scramble=True)
        draws = sobol.draw(n).to(dtype=torch.double)
        # Pin masked-out dimensions to 0.5 so they don't perturb the GP.
        for idx, active in enumerate(self._active_mask):
            if not active:
                draws[:, idx] = 0.5
        return draws

    def _x_to_params(self, x) -> Dict[str, float]:
        if x.ndim != 1 or x.shape[0] != len(PARAMETER_ORDER):
            raise ValueError("x must be a 1D tensor with len(PARAMETER_ORDER) dimensions")
        params: Dict[str, float] = {}
        for idx, name in enumerate(PARAMETER_ORDER):
            bmin, bmax = self._bounds[name]
            params[name] = float(bmin + float(x[idx]) * (bmax - bmin))
        # Backward-compatible alias; the CFD pipeline reads Q_total.
        params["Q"] = params["Q_total"]
        return params

    def _evaluate_point(self, x) -> Tuple[float, List[float]]:
        from ooc_optimizer.cfd import evaluate_cfd

        params = self._x_to_params(x)
        start = time.perf_counter()
        metrics = evaluate_cfd(
            params=params,
            pillar_config=self.pillar_config,
            H_um=self.H,
            config=self._cfd_config,
            topology=self.topology,
            target_profile=self.target_profile,
        )
        wall = time.perf_counter() - start

        for key in ("L2_to_target", "tau_mean", "f_dead"):
            if key not in metrics:
                raise ValueError(f"evaluate_cfd returned incomplete metrics (missing {key})")

        raw_L2 = float(metrics["L2_to_target"])
        if not (raw_L2 == raw_L2):  # NaN check
            raw_L2 = self.penalty_L2
        mesh_ok = bool(metrics.get("mesh_ok", True))
        converged = bool(metrics.get("converged", False))
        objective = raw_L2 if (mesh_ok and converged) else self.penalty_L2

        c1 = float(metrics["tau_mean"] - self._constraint_config["tau_mean_min"])
        c2 = float(self._constraint_config["tau_mean_max"] - metrics["tau_mean"])
        c3 = float(self._constraint_config["f_dead_max"] - metrics["f_dead"])

        record = {
            "params": params,
            "metrics": metrics,
            "objective": objective,
            "raw_L2": raw_L2,
            "constraints": [c1, c2, c3],
            "feasible": bool(
                c1 >= 0 and c2 >= 0 and c3 >= 0 and converged and mesh_ok
            ),
            "wall_time_s": wall,
        }
        self.evaluations.append(record)
        self.logger.log_evaluation(
            params=params,
            pillar_config=self.pillar_config,
            H=self.H,
            inlet_topology=self.topology,
            target_profile={"kind": self.target_profile.kind, **self.target_profile.params},
            metrics=metrics,
            wall_time_s=wall,
            case_dir=Path(metrics["case_dir"]) if metrics.get("case_dir") else None,
        )
        return objective, [c1, c2, c3]

    def _fit_gp(self) -> None:
        if self.train_X is None or self.train_Y is None or self.train_constraints is None:
            raise ValueError("Training data is empty; cannot fit GP models")
        deps = self._require_bo_stack()
        SingleTaskGP = deps["SingleTaskGP"]
        Standardize = deps["Standardize"]
        ExactMarginalLogLikelihood = deps["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = deps["fit_gpytorch_mll"]

        self._objective_model = SingleTaskGP(
            self.train_X, self.train_Y, outcome_transform=Standardize(m=1)
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self._objective_model.likelihood, self._objective_model))

        self._constraint_models = []
        for idx in range(3):
            target = self.train_constraints[:, idx].unsqueeze(-1)
            model = SingleTaskGP(self.train_X, target, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            self._constraint_models.append(model)

    def _best_observed_feasible_objective(self) -> float:
        deps = self._require_bo_stack()
        torch = deps["torch"]
        mask = torch.all(self.train_constraints >= 0, dim=1)
        if torch.any(mask):
            return float(torch.min(self.train_Y[mask]).item())
        return float(torch.min(self.train_Y).item())

    def _constrained_expected_improvement(self, x):
        if self._objective_model is None or len(self._constraint_models) != 3:
            raise ValueError("Models are not fitted; call _fit_gp first")
        deps = self._require_bo_stack()
        torch = deps["torch"]
        Normal = deps["Normal"]

        x_batch = x.unsqueeze(0)
        posterior = self._objective_model.posterior(x_batch)
        mu = posterior.mean.squeeze(-1).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)

        best_f = self._best_observed_feasible_objective()
        z = (best_f - mu) / sigma
        normal = Normal(torch.tensor(0.0, dtype=torch.double), torch.tensor(1.0, dtype=torch.double))
        ei = (best_f - mu) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
        ei = torch.clamp(ei, min=0.0)

        prob_feas = torch.tensor(1.0, dtype=torch.double)
        for model in self._constraint_models:
            c_post = model.posterior(x_batch)
            c_mu = c_post.mean.squeeze(-1).squeeze(-1)
            c_sigma = c_post.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
            prob_feas = prob_feas * normal.cdf(c_mu / c_sigma)
        return ei * prob_feas

    def _optimize_acquisition(self):
        deps = self._require_bo_stack()
        torch = deps["torch"]
        candidates = self._generate_sobol_points(N_ACQ_SAMPLES)
        acq_values = []
        for x in candidates:
            acq_values.append(self._constrained_expected_improvement(x))
        acq_tensor = torch.stack(acq_values)
        best_idx = int(torch.argmax(acq_tensor).item())
        return candidates[best_idx]

    def _get_best_feasible(self) -> Optional[Dict]:
        feasible = [r for r in self.evaluations if r["feasible"]]
        if not feasible:
            return None
        best = min(feasible, key=lambda r: r["objective"])
        return {
            "params": best["params"],
            "L2_to_target": best["objective"],
            "metrics": best["metrics"],
            "wall_time_s": best["wall_time_s"],
        }

    def save_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "topology": self.topology,
            "pillar_config": self.pillar_config,
            "H": self.H,
            "parameter_order": PARAMETER_ORDER,
            "active_mask": self._active_mask,
            "bounds": self._bounds,
            "constraints": self._constraint_config,
            "target_profile": {"kind": self.target_profile.kind, **self.target_profile.params},
            "evaluations": self.evaluations,
            "train_X": self.train_X.tolist() if self.train_X is not None else None,
            "train_Y": self.train_Y.squeeze(-1).tolist() if self.train_Y is not None else None,
            "train_constraints": (
                self.train_constraints.tolist() if self.train_constraints is not None else None
            ),
        }
        with open(output_dir / "evaluations.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

        model_states: Dict[str, Any] = {}
        if self._objective_model is not None:
            model_states["objective"] = self._objective_model.state_dict()
        for idx, model in enumerate(self._constraint_models):
            model_states[f"constraint_{idx + 1}"] = model.state_dict()
        if model_states:
            deps = self._require_bo_stack()
            torch = deps["torch"]
            torch.save(model_states, output_dir / "gp_model_state.pt")
