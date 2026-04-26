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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ooc_optimizer.config.logger import EvaluationLogger
from ooc_optimizer.optimization.objectives import TargetProfile, build_target_profile

logger = logging.getLogger(__name__)

N_SOBOL_INIT = 20
N_BO_ITERATIONS = 40
# Acquisition optimisation: L-BFGS multistart from Sobol-seeded initial points.
# RAW_SAMPLES seeds, NUM_RESTARTS L-BFGS runs from the top scoring seeds.
ACQ_RAW_SAMPLES = 1024
ACQ_NUM_RESTARTS = 16
# GP noise floor: prevents the kernel from interpolating near-duplicate points
# in flat regions of the response surface, which previously inflated the
# total-effect Sobol indices in the interpretability stage.
GP_NOISE_FLOOR = 1.0e-4

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
    if topology == "ladder":
        # Ladder is a fixed N-strip y-stacked-inlet topology with equal flow
        # split; only chamber width W and Q_total influence the field. Pin the
        # rest to midpoint via the mask so the GP doesn't waste effort on
        # informationless inputs.
        for name in ("theta", "r_flow", "delta_W", "d_p", "s_p"):
            active[PARAMETER_ORDER.index(name)] = False
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
        if topology not in {"opposing", "same_side_Y", "asymmetric_lumen", "ladder"}:
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
        self.batch_size = int(config["optimization"].get("batch_size", 1))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {self.batch_size})")
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
        # Lock guarding mutations to shared state from concurrent CFD threads.
        self._eval_lock = threading.Lock()
        self._objective_model = None
        self._constraint_models: List[Any] = []
        self._model_list = None
        self._bo_deps: Optional[Dict[str, Any]] = None

    # --------------------------- dependency handling -----------------------

    def _require_bo_stack(self) -> Dict[str, Any]:
        if self._bo_deps is not None:
            return self._bo_deps
        try:
            import torch
            from botorch.acquisition.analytic import ConstrainedExpectedImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import ModelListGP, SingleTaskGP
            from botorch.models.transforms.outcome import Standardize
            from botorch.optim import optimize_acqf
            from gpytorch.constraints import GreaterThan
            from gpytorch.likelihoods import GaussianLikelihood
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from gpytorch.priors import GammaPrior
            from torch.distributions import Normal
            from torch.quasirandom import SobolEngine
        except ImportError as exc:
            raise ImportError(
                "BO dependencies missing. Install requirements.txt (torch, gpytorch, botorch)."
            ) from exc
        self._bo_deps = {
            "torch": torch,
            "ConstrainedExpectedImprovement": ConstrainedExpectedImprovement,
            "fit_gpytorch_mll": fit_gpytorch_mll,
            "ModelListGP": ModelListGP,
            "SingleTaskGP": SingleTaskGP,
            "Standardize": Standardize,
            "optimize_acqf": optimize_acqf,
            "GreaterThan": GreaterThan,
            "GaussianLikelihood": GaussianLikelihood,
            "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
            "GammaPrior": GammaPrior,
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
        required = (
            "tau_mean_min", "tau_mean_max", "f_dead_max",
            "Re_max", "aspect_ratio_max",
        )
        for key in required:
            if key not in c:
                raise ValueError(f"Missing constraint key '{key}'")
        return {k: float(c[k]) for k in required}

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
        """Execute Sobol init + BO iterations; return full results dict.

        When ``optimization.batch_size > 1`` both the Sobol initialisation
        and each BO outer iteration evaluate ``batch_size`` candidates
        concurrently via ``_evaluate_batch``. Total evaluation count is
        unchanged — only wall-clock latency drops.
        """
        logger.info(
            "Starting BO: topology=%s, pillar=%s, H=%.0f (Sobol=%d, BO=%d, batch=%d, active_dim=%d)",
            self.topology, self.pillar_config, self.H,
            self.n_sobol, self.n_bo, self.batch_size, self._n_active,
        )
        deps = self._require_bo_stack()
        torch = deps["torch"]

        init_points = self._generate_sobol_points(self.n_sobol)
        X_rows: List[Any] = []
        Y_rows: List[float] = []
        C_rows: List[List[float]] = []
        # Batched Sobol init: chunk the Sobol draws and farm each chunk to the
        # thread pool so up to `batch_size` CFD cases run concurrently.
        for chunk_start in range(0, self.n_sobol, self.batch_size):
            chunk = [init_points[i] for i in range(chunk_start, min(chunk_start + self.batch_size, self.n_sobol))]
            results = self._evaluate_batch(chunk)
            for x, (objective, constraints) in zip(chunk, results):
                X_rows.append(x)
                Y_rows.append(objective)
                C_rows.append(constraints)

        self.train_X = torch.stack(X_rows).to(dtype=torch.double)
        self.train_Y = torch.tensor(Y_rows, dtype=torch.double).unsqueeze(-1)
        self.train_constraints = torch.tensor(C_rows, dtype=torch.double)

        # Batched BO loop: each outer round fits the GP once on all data so
        # far, then proposes and concurrently evaluates `batch_size` greedy
        # candidates. Total evaluations stay at exactly self.n_bo.
        evals_done = 0
        round_idx = 0
        while evals_done < self.n_bo:
            self._fit_gp()
            q = min(self.batch_size, self.n_bo - evals_done)
            candidates = self._optimize_acquisition(q=q)
            results = self._evaluate_batch(candidates)

            X_new = torch.stack(candidates).to(dtype=torch.double)
            Y_new = torch.tensor([[r[0]] for r in results], dtype=torch.double)
            C_new = torch.tensor([r[1] for r in results], dtype=torch.double)
            self.train_X = torch.cat([self.train_X, X_new], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_new], dim=0)
            self.train_constraints = torch.cat([self.train_constraints, C_new], dim=0)

            round_idx += 1
            evals_done += q
            best_in_round = min(r[0] for r in results)
            logger.info(
                "BO round %d (q=%d, %d/%d evals, %s, %s, H=%.0f): best L2 in round=%.4f",
                round_idx, q, evals_done, self.n_bo,
                self.topology, self.pillar_config, self.H, best_in_round,
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

        for key in ("L2_to_target", "tau_mean", "f_dead", "Re", "aspect_ratio"):
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
        # Laminar + aspect-ratio gates.  Any NaN that slipped through the
        # PENALTY_METRICS / metrics-fallback paths is mapped to a large finite
        # sentinel (not inf) so the constraint slack stays a finite negative
        # number — BoTorch's ModelListGP refuses to fit on -inf or NaN.
        Re_val = float(metrics["Re"])
        if not (Re_val == Re_val):  # NaN
            Re_val = 1.0e6
        c4 = float(self._constraint_config["Re_max"] - Re_val)
        ar_val = float(metrics["aspect_ratio"])
        if not (ar_val == ar_val):
            ar_val = 1.0e3
        c5 = float(self._constraint_config["aspect_ratio_max"] - ar_val)

        record = {
            "params": params,
            "metrics": metrics,
            "objective": objective,
            "raw_L2": raw_L2,
            "constraints": [c1, c2, c3, c4, c5],
            "feasible": bool(
                c1 >= 0 and c2 >= 0 and c3 >= 0 and c4 >= 0 and c5 >= 0
                and converged and mesh_ok
            ),
            "wall_time_s": wall,
        }
        # Mutations to evaluations list and the JSONL logger must be serialized
        # when _evaluate_point is invoked from a ThreadPoolExecutor batch.
        with self._eval_lock:
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
        return objective, [c1, c2, c3, c4, c5]

    def _evaluate_batch(self, points: List[Any]) -> List[Tuple[float, List[float]]]:
        """Evaluate a list of candidate points concurrently.

        Each OpenFOAM case writes to its own timestamped directory, so the only
        shared mutable state is `self.evaluations` and the JSONL logger — both
        guarded by `self._eval_lock`. Threads (rather than processes) are used
        because `evaluate_cfd` shells out to OpenFOAM via subprocess, which
        releases the GIL and avoids pickling overhead inside the outer
        ProcessPoolExecutor that drives `BORunner` instances.
        """
        if not points:
            return []
        if self.batch_size <= 1 or len(points) == 1:
            return [self._evaluate_point(x) for x in points]

        results: List[Optional[Tuple[float, List[float]]]] = [None] * len(points)
        max_workers = min(self.batch_size, len(points))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(self._evaluate_point, x): i for i, x in enumerate(points)}
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                results[idx] = fut.result()
        # Order is preserved; cast away Optional now that all slots are filled.
        return [r for r in results if r is not None]

    def _fit_gp(self) -> None:
        if self.train_X is None or self.train_Y is None or self.train_constraints is None:
            raise ValueError("Training data is empty; cannot fit GP models")
        deps = self._require_bo_stack()
        SingleTaskGP = deps["SingleTaskGP"]
        Standardize = deps["Standardize"]
        ExactMarginalLogLikelihood = deps["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = deps["fit_gpytorch_mll"]
        ModelListGP = deps["ModelListGP"]
        GaussianLikelihood = deps["GaussianLikelihood"]
        GreaterThan = deps["GreaterThan"]
        GammaPrior = deps["GammaPrior"]

        def _build_gp(X, Y):
            # Wider noise prior + a hard floor on the noise variance keeps the
            # kernel from interpolating coincident or near-coincident points
            # in flat regions; this is what produces ΣS_T » 1 downstream.
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(1.1, 0.05),
                noise_constraint=GreaterThan(GP_NOISE_FLOOR),
            )
            model = SingleTaskGP(
                X, Y,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            return model

        self._objective_model = _build_gp(self.train_X, self.train_Y)
        self._constraint_models = [
            _build_gp(self.train_X, self.train_constraints[:, idx].unsqueeze(-1))
            for idx in range(5)
        ]
        self._model_list = ModelListGP(self._objective_model, *self._constraint_models)

    def _best_observed_feasible_objective(self) -> float:
        deps = self._require_bo_stack()
        torch = deps["torch"]
        mask = torch.all(self.train_constraints >= 0, dim=1)
        if torch.any(mask):
            return float(torch.min(self.train_Y[mask]).item())
        return float(torch.min(self.train_Y).item())

    def _optimize_acquisition(self, q: int = 1):
        """L-BFGS multistart maximisation of constrained-EI on the joint GP.

        The previous implementation scored 2048 random Sobol candidates and
        picked the argmax — this is no better than random search once the
        feasible region narrows around an optimum.  We now use BoTorch's
        ``optimize_acqf`` (Sobol seeding → top-k → L-BFGS-B refinement) on
        the analytical ``ConstrainedExpectedImprovement`` defined over a
        joint ``ModelListGP`` of (objective, c1, c2, c3).  Inactive
        parameters are pinned to 0.5 via ``fixed_features`` so the gradient
        optimiser never wastes effort on informationless dimensions.

        When ``q > 1`` the analytic CEI is sequentially greedy-batched via
        ``optimize_acqf(sequential=True)``, returning ``q`` candidates that
        are evaluated concurrently by the caller.
        """
        if q < 1:
            raise ValueError(f"q must be >= 1 (got {q})")
        if self._model_list is None:
            raise ValueError("Models are not fitted; call _fit_gp first")
        deps = self._require_bo_stack()
        torch = deps["torch"]
        ConstrainedExpectedImprovement = deps["ConstrainedExpectedImprovement"]
        optimize_acqf = deps["optimize_acqf"]

        best_f = self._best_observed_feasible_objective()
        # Constraint slacks are encoded as c >= 0; in the joint model
        # output indices 1..5 are the five constraint GPs (tau_mean lower/upper,
        # f_dead, Re_max, aspect_ratio_max).
        acq = ConstrainedExpectedImprovement(
            model=self._model_list,
            best_f=best_f,
            objective_index=0,
            constraints={
                1: (0.0, None), 2: (0.0, None), 3: (0.0, None),
                4: (0.0, None), 5: (0.0, None),
            },
            maximize=False,  # objective is L2 (minimised)
        )

        d = len(PARAMETER_ORDER)
        bounds = torch.stack(
            [torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)]
        )
        fixed_features = {
            idx: 0.5 for idx, active in enumerate(self._active_mask) if not active
        }

        # Analytic ConstrainedExpectedImprovement in BoTorch 0.12 does not
        # implement the X_pending hook needed by sequential greedy q-batches.
        # For q == 1 we call optimize_acqf once.  For q > 1 we run q
        # independent L-BFGS multistarts and rely on the stochastic Sobol
        # seeding to diversify them.  This is not full qEI (no fantasy
        # update between picks), but it preserves the L-BFGS quality of the
        # acquisition optimisation, which was the actual prior bottleneck.
        def _single_pick():
            cand_q1, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=ACQ_NUM_RESTARTS,
                raw_samples=ACQ_RAW_SAMPLES,
                fixed_features=fixed_features if fixed_features else None,
                options={"batch_limit": 5, "maxiter": 200},
            )
            return cand_q1.squeeze(0).to(dtype=torch.double)

        candidates = [_single_pick() for _ in range(q)]
        return candidates

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
