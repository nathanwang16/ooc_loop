"""
Structured evaluation logger.

Appends every CFD evaluation to a JSONL file for reproducibility,
convergence tracking, and later analysis.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EvaluationLogger:
    """Append-only structured log of CFD evaluations.

    Each entry records: input parameters, discrete configuration, output
    metrics, wall-clock time, convergence status, and case directory.
    """

    def __init__(self, log_path: Path):
        """
        Parameters
        ----------
        log_path : Path
            Path to the JSONL log file (created if it doesn't exist).

        Raises
        ------
        ValueError
            If log_path is None or empty.
        """
        if log_path is None:
            raise ValueError("log_path must not be None")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Evaluation logger initialized at %s", self.log_path)

    def log_evaluation(
        self,
        params: Dict[str, float],
        pillar_config: str,
        H: float,
        metrics: Dict[str, Any],
        wall_time_s: float,
        case_dir: Optional[Path] = None,
    ) -> None:
        """Append one evaluation record to the log file."""
        record = {
            "timestamp": time.time(),
            "params": params,
            "pillar_config": pillar_config,
            "H": H,
            "metrics": metrics,
            "wall_time_s": wall_time_s,
            "case_dir": str(case_dir) if case_dir else None,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_all(self):
        """Read all evaluation records from the log file.

        Returns
        -------
        records : list of dict
        """
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_best_feasible(self, tau_range=(0.5, 2.0), max_f_dead=0.05):
        """Find the evaluation with the lowest CV(τ) among feasible results.

        Returns
        -------
        best_record : dict or None
        """
        records = self.load_all()
        feasible = [
            r for r in records
            if r["metrics"].get("converged", False)
            and r["metrics"].get("mesh_ok", True)
            and tau_range[0] <= r["metrics"].get("tau_mean", 0) <= tau_range[1]
            and r["metrics"].get("f_dead", 1.0) <= max_f_dead
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda r: r["metrics"]["cv_tau"])
