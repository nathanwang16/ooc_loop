"""
Module 2.3 — Structured evaluation logger (v2).

Each line of the JSONL file is a full evaluation record including the v2
fields needed by Modules 3.1–3.3 and 4.x:

    timestamp, params, pillar_config, H, inlet_topology, target_profile,
    metrics (L2_to_target, tau_mean, f_dead, grad_sharpness, monotonicity,
    converged_U, converged_C, ...), wall_time_s, case_dir.

The log is append-only; corresponding replay utilities (``load_all``,
``get_best_feasible``, ``filter_by_topology``) support downstream analysis
without rerunning CFD.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class EvaluationLogger:
    """Append-only structured log of CFD evaluations (v2 schema)."""

    def __init__(self, log_path: Path):
        if log_path is None:
            raise ValueError("log_path must not be None")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Evaluation logger initialised at %s", self.log_path)

    def log_evaluation(
        self,
        *,
        params: Dict[str, float],
        pillar_config: str,
        H: float,
        metrics: Dict[str, Any],
        wall_time_s: float,
        case_dir: Optional[Path] = None,
        inlet_topology: Optional[str] = None,
        target_profile: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "params": params,
            "pillar_config": pillar_config,
            "H": H,
            "inlet_topology": inlet_topology,
            "target_profile": target_profile,
            "metrics": metrics,
            "wall_time_s": wall_time_s,
            "case_dir": str(case_dir) if case_dir else None,
        }
        if extra:
            record.update(extra)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def filter_by_topology(self, topology: str) -> List[Dict[str, Any]]:
        return [r for r in self.load_all() if r.get("inlet_topology") == topology]

    def get_best_feasible(
        self,
        *,
        objective_key: Optional[str] = None,
        tau_range=(0.1, 2.0),
        max_f_dead: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Return the record minimizing ``objective_key`` among feasible entries.

        When ``objective_key`` is None, prefer ``L2_to_target`` (v2 objective)
        and fall back to ``cv_tau`` (v1 objective) when the former is absent
        or NaN in the first record of the log.  This keeps the v1 WSS-example
        workflow functional alongside the v2 one.
        """
        records = self.load_all()
        if not records:
            return None
        if objective_key is None:
            first_metrics = records[0].get("metrics", {})
            if "L2_to_target" in first_metrics and _is_finite(first_metrics.get("L2_to_target")):
                objective_key = "L2_to_target"
            else:
                objective_key = "cv_tau"
        feasible: Iterable[Dict[str, Any]] = (
            r for r in records
            if r.get("metrics", {}).get("converged", False)
            and r.get("metrics", {}).get("mesh_ok", True)
            and tau_range[0] <= r["metrics"].get("tau_mean", 0) <= tau_range[1]
            and r["metrics"].get("f_dead", 1.0) <= max_f_dead
        )
        best = None
        best_val = float("inf")
        for r in feasible:
            val = r["metrics"].get(objective_key)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v < best_val:
                best_val = v
                best = r
        return best


def _is_finite(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _json_default(obj: Any) -> Any:
    """Coerce Path / numpy scalars / bools to JSON-serialisable types."""
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Type {type(obj).__name__} not JSON serialisable")
