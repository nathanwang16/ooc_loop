"""
Bayesian optimization runner for a single discrete configuration.

Uses BoTorch with:
    - Matérn 5/2 GP surrogate
    - 15-point Sobol initialization
    - ConstrainedExpectedImprovement acquisition
    - 35 BO iterations (50 total evaluations)

Constraints:
    c1 = τ_mean - 0.5     (feasible if ≥ 0)
    c2 = 2.0 - τ_mean     (feasible if ≥ 0)
    c3 = 0.05 - f_dead    (feasible if ≥ 0)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ooc_optimizer.config.logger import EvaluationLogger

logger = logging.getLogger(__name__)

N_SOBOL_INIT = 15
N_BO_ITERATIONS = 35
N_ACQ_SAMPLES = 2048
PARAMETER_ORDER = ["W", "d_p", "s_p", "theta", "Q"]


class BORunner:
    """Bayesian optimization loop for one (pillar_config, H) combination.

    Parameters
    ----------
    config : dict
        Loaded configuration with bounds, solver settings, paths.
    pillar_config : str
        One of {"none", "1x4", "2x4", "3x6"}.
    H : float
        Chamber height in μm.
    """

    def __init__(self, config: dict, pillar_config: str, H: float):
        if config is None:
            raise ValueError("config must not be None")
        if pillar_config not in {"none", "1x4", "2x4", "3x6"}:
            raise ValueError(f"Invalid pillar_config: {pillar_config}")
        if H not in (200.0, 300.0):
            raise ValueError(f"Invalid chamber height: {H}")
        if "continuous_bounds" not in config:
            raise ValueError("Missing 'continuous_bounds' in config")
        if "optimization" not in config:
            raise ValueError("Missing 'optimization' section in config")
        if "paths" not in config:
            raise ValueError("Missing 'paths' section in config")

        self.config = config
        self.pillar_config = pillar_config
        self.H = H

        self._bounds = self._extract_bounds(config["continuous_bounds"])
        self._constraint_config = self._extract_constraint_config(config["optimization"])
        self.n_sobol = int(config["optimization"]["n_sobol_init"])
        self.n_bo = int(config["optimization"]["n_bo_iterations"])
        self.penalty_cv_tau = float(config["optimization"]["penalty_cv_tau"])
        self.results_dir = Path(config["paths"]["results_dir"])

        self._cfd_config = self._build_cfd_config(config)
        self.logger = self._build_evaluation_logger(config)

        self.train_X = None
        self.train_Y = None
        self.train_constraints = None
        self.evaluations: List[Dict[str, Any]] = []
        self._objective_model = None
        self._constraint_models: List[Any] = []
        self._bo_deps: Optional[Dict[str, Any]] = None

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
                "BO dependencies are missing. Install requirements.txt "
                "(torch, gpytorch, botorch) before running optimization."
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
        constraints = opt_cfg["constraints"]
        required = ("tau_mean_min", "tau_mean_max", "f_dead_max")
        missing = [key for key in required if key not in constraints]
        if missing:
            raise ValueError(f"Missing constraint keys: {missing}")
        return {
            "tau_mean_min": float(constraints["tau_mean_min"]),
            "tau_mean_max": float(constraints["tau_mean_max"]),
            "f_dead_max": float(constraints["f_dead_max"]),
        }

    @staticmethod
    def _build_cfd_config(config: Dict[str, Any]) -> Dict[str, Any]:
        paths = config["paths"]
        required = ("case_output_dir", "template_case")
        missing = [k for k in required if k not in paths]
        if missing:
            raise ValueError(f"Missing path keys required for CFD: {missing}")

        cfd_config = dict(config)
        cfd_paths = dict(paths)
        cfd_paths["work_dir"] = paths["case_output_dir"]
        cfd_paths["template_dir"] = paths["template_case"]
        cfd_config["paths"] = cfd_paths

        fixed = config.get("fixed_parameters")
        if fixed is None or "fluid_viscosity_Pa_s" not in fixed:
            raise ValueError("Missing fixed_parameters.fluid_viscosity_Pa_s")
        cfd_config["physics"] = {"mu": float(fixed["fluid_viscosity_Pa_s"])}
        return cfd_config

    def _build_evaluation_logger(self, config: Dict[str, Any]) -> EvaluationLogger:
        eval_log_path = Path(config["paths"]["evaluation_log"])
        stem = eval_log_path.stem
        suffix = eval_log_path.suffix or ".jsonl"
        scoped_name = f"{stem}_{self.pillar_config}_H{int(self.H)}{suffix}"
        scoped_log = eval_log_path.with_name(scoped_name)
        return EvaluationLogger(scoped_log)

    def run(self) -> Dict:
        """Execute the full BO loop: Sobol init + BO iterations.

        Returns
        -------
        results : dict
            Contains best_params, best_cv_tau, all evaluations, GP model state.
        """
        logger.info(
            "Starting BO for config=%s, H=%.0f μm (%d Sobol + %d BO)",
            self.pillar_config,
            self.H,
            self.n_sobol,
            self.n_bo,
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
                [self.train_Y, torch.tensor([[objective]], dtype=torch.double)],
                dim=0,
            )
            self.train_constraints = torch.cat(
                [self.train_constraints, torch.tensor([constraints], dtype=torch.double)],
                dim=0,
            )
            logger.info(
                "BO iter %d/%d complete for %s H=%.0f; cv_tau=%.4f",
                iteration + 1,
                self.n_bo,
                self.pillar_config,
                self.H,
                objective,
            )

        best = self._get_best_feasible()
        state_dir = self.results_dir / f"bo_{self.pillar_config}_H{int(self.H)}"
        self.save_state(state_dir)

        result = {
            "config_name": f"{self.pillar_config}_H{int(self.H)}",
            "pillar_config": self.pillar_config,
            "H": self.H,
            "n_evaluations": len(self.evaluations),
            "best_feasible": best,
            "best_cv_tau": best["cv_tau"] if best else None,
            "best_params": best["params"] if best else None,
            "state_dir": str(state_dir),
            "evaluations": self.evaluations,
        }
        return result

    def _generate_sobol_points(self, n: int = N_SOBOL_INIT):
        """Generate quasi-random initial points within parameter bounds."""
        if n <= 0:
            raise ValueError("n must be > 0 for Sobol generation")
        deps = self._require_bo_stack()
        SobolEngine = deps["SobolEngine"]
        torch = deps["torch"]
        sobol = SobolEngine(dimension=len(PARAMETER_ORDER), scramble=True)
        return sobol.draw(n).to(dtype=torch.double)

    def _x_to_params(self, x) -> Dict[str, float]:
        if x.ndim != 1 or x.shape[0] != len(PARAMETER_ORDER):
            raise ValueError("x must be a 1D tensor with 5 dimensions")
        params: Dict[str, float] = {}
        for idx, name in enumerate(PARAMETER_ORDER):
            bmin, bmax = self._bounds[name]
            params[name] = float(bmin + float(x[idx]) * (bmax - bmin))
        return params

    def _evaluate_point(self, x) -> Tuple[float, List[float]]:
        """Call the CFD pipeline for one parameter vector.

        Returns (objective, [c1, c2, c3]) tuple.
        """
        from ooc_optimizer.cfd import evaluate_cfd

        params = self._x_to_params(x)
        start = time.perf_counter()
        metrics = evaluate_cfd(
            params=params,
            pillar_config=self.pillar_config,
            H_um=self.H,
            config=self._cfd_config,
        )
        wall_time_s = time.perf_counter() - start

        if "cv_tau" not in metrics or "tau_mean" not in metrics or "f_dead" not in metrics:
            raise ValueError("evaluate_cfd returned incomplete metrics")

        raw_objective = float(metrics["cv_tau"])
        mesh_ok = bool(metrics.get("mesh_ok", True))
        objective = raw_objective if mesh_ok else self.penalty_cv_tau
        c1 = float(metrics["tau_mean"] - self._constraint_config["tau_mean_min"])
        c2 = float(self._constraint_config["tau_mean_max"] - metrics["tau_mean"])
        c3 = float(self._constraint_config["f_dead_max"] - metrics["f_dead"])

        record = {
            "params": params,
            "metrics": metrics,
            "objective": objective,
            "raw_objective": raw_objective,
            "constraints": [c1, c2, c3],
            "feasible": bool(
                c1 >= 0
                and c2 >= 0
                and c3 >= 0
                and metrics.get("converged", False)
                and mesh_ok
            ),
            "wall_time_s": wall_time_s,
        }
        self.evaluations.append(record)

        self.logger.log_evaluation(
            params=params,
            pillar_config=self.pillar_config,
            H=self.H,
            metrics=metrics,
            wall_time_s=wall_time_s,
            case_dir=Path(metrics["case_dir"]) if metrics.get("case_dir") else None,
        )
        return objective, [c1, c2, c3]

    def _fit_gp(self) -> None:
        """Fit/refit the GP surrogate and constraint GPs."""
        if self.train_X is None or self.train_Y is None or self.train_constraints is None:
            raise ValueError("Training data is empty; cannot fit GP models")
        deps = self._require_bo_stack()
        SingleTaskGP = deps["SingleTaskGP"]
        Standardize = deps["Standardize"]
        ExactMarginalLogLikelihood = deps["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = deps["fit_gpytorch_mll"]

        self._objective_model = SingleTaskGP(
            self.train_X,
            self.train_Y,
            outcome_transform=Standardize(m=1),
        )
        objective_mll = ExactMarginalLogLikelihood(
            self._objective_model.likelihood,
            self._objective_model,
        )
        fit_gpytorch_mll(objective_mll)

        self._constraint_models = []
        for idx in range(3):
            target = self.train_constraints[:, idx].unsqueeze(-1)
            model = SingleTaskGP(
                self.train_X,
                target,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self._constraint_models.append(model)

    def _best_observed_feasible_cv(self) -> float:
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
        objective_posterior = self._objective_model.posterior(x_batch)
        mu = objective_posterior.mean.squeeze(-1).squeeze(-1)
        sigma = objective_posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)

        best_f = self._best_observed_feasible_cv()
        z = (best_f - mu) / sigma
        normal = Normal(
            torch.tensor(0.0, dtype=torch.double),
            torch.tensor(1.0, dtype=torch.double),
        )
        ei = (best_f - mu) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
        ei = torch.clamp(ei, min=0.0)

        prob_feas = torch.tensor(1.0, dtype=torch.double)
        for model in self._constraint_models:
            posterior = model.posterior(x_batch)
            c_mu = posterior.mean.squeeze(-1).squeeze(-1)
            c_sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
            z_c = c_mu / c_sigma
            p = normal.cdf(z_c)
            prob_feas = prob_feas * p

        return ei * prob_feas

    def _optimize_acquisition(self):
        """Maximize ConstrainedExpectedImprovement to get next candidate."""
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
        """Return the best feasible point found so far."""
        feasible = [record for record in self.evaluations if record["feasible"]]
        if not feasible:
            return None
        best = min(feasible, key=lambda r: r["objective"])
        return {
            "params": best["params"],
            "cv_tau": best["objective"],
            "metrics": best["metrics"],
            "wall_time_s": best["wall_time_s"],
        }

    def save_state(self, output_dir: Path) -> None:
        """Serialize GP model, training data, and evaluation log."""
        output_dir.mkdir(parents=True, exist_ok=True)

        data_payload = {
            "pillar_config": self.pillar_config,
            "H": self.H,
            "parameter_order": PARAMETER_ORDER,
            "bounds": self._bounds,
            "constraints": self._constraint_config,
            "evaluations": self.evaluations,
            "train_X": self.train_X.tolist() if self.train_X is not None else None,
            "train_Y": self.train_Y.squeeze(-1).tolist() if self.train_Y is not None else None,
            "train_constraints": (
                self.train_constraints.tolist() if self.train_constraints is not None else None
            ),
        }
        with open(output_dir / "evaluations.json", "w") as f:
            json.dump(data_payload, f, indent=2)

        model_states = {}
        if self._objective_model is not None:
            model_states["objective"] = self._objective_model.state_dict()
        for idx, model in enumerate(self._constraint_models):
            model_states[f"constraint_{idx + 1}"] = model.state_dict()
        if model_states:
            deps = self._require_bo_stack()
            torch = deps["torch"]
            torch.save(model_states, output_dir / "gp_model_state.pt")
