"""Smoke test for intra-config BO trial batching.

Patches ``ooc_optimizer.cfd.evaluate_cfd`` with a fast stub so we exercise
the threaded evaluation path without invoking OpenFOAM.  The stub records
(start_time, thread_name) for every call; we then assert that with
``batch_size > 1`` multiple distinct threads are observed running CFD
calls concurrently, and that the locked evaluations list has the right
length and no missing/duplicated entries.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401  (only to skip cleanly if BO stack absent)
    from ooc_optimizer.optimization.bo_loop import BORunner
except ImportError:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="BO stack (torch/botorch) not installed")


def _make_config(tmp_path: Path, *, batch_size: int, n_sobol: int, n_bo: int) -> dict:
    """Minimal in-memory config sufficient to construct a BORunner."""
    return {
        "continuous_bounds": {
            "W":       {"min": 1500.0, "max": 4500.0},
            "d_p":     {"min": 50.0,   "max": 300.0},
            "s_p":     {"min": 100.0,  "max": 400.0},
            "theta":   {"min": 0.0,    "max": 30.0},
            "Q_total": {"min": 10.0,   "max": 200.0},
            "r_flow":  {"min": 0.10,   "max": 0.97},
            "delta_W": {"min": 0.12,   "max": 0.48},
        },
        "optimization": {
            "n_sobol_init": n_sobol,
            "n_bo_iterations": n_bo,
            "batch_size": batch_size,
            "max_concurrent_cfd": 8,
            "constraints": {
                "tau_mean_min": 0.1,
                "tau_mean_max": 5.0,
                "f_dead_max": 0.05,
            },
            "penalty_L2": 99.0,
        },
        "paths": {
            "results_dir":     str(tmp_path / "results"),
            "case_output_dir": str(tmp_path / "cases"),
            "template_case":   str(tmp_path / "template"),
            "evaluation_log":  str(tmp_path / "results" / "eval.jsonl"),
        },
        "fixed_parameters": {"fluid_viscosity_Pa_s": 1.0e-3},
        "diffusivity": 1.0e-10,
        "target_profile": {"kind": "linear_gradient"},
    }


def _stub_evaluate_cfd_factory(call_log: list, lock: threading.Lock, sleep_s: float = 0.30):
    """Return a stub that mimics evaluate_cfd's contract for BORunner."""
    def stub(**kwargs):
        with lock:
            call_log.append((time.perf_counter(), threading.current_thread().name))
        # Simulate the time OpenFOAM would spend in subprocess.run (GIL released).
        time.sleep(sleep_s)
        return {
            "L2_to_target": 0.42,
            "tau_mean": 0.8,
            "f_dead": 0.01,
            "mesh_ok": True,
            "converged": True,
            "case_dir": None,
        }
    return stub


@pytest.mark.parametrize(
    ("batch_size", "expected_min_concurrency"),
    [(1, 1), (4, 2)],  # batch=4 must show >=2 overlapping calls
)
def test_batched_run_dispatches_concurrently(tmp_path, monkeypatch, batch_size, expected_min_concurrency):
    """run() with batch_size>1 evaluates multiple CFD calls concurrently."""
    n_sobol, n_bo = 4, 4
    cfg = _make_config(tmp_path, batch_size=batch_size, n_sobol=n_sobol, n_bo=n_bo)

    call_log: list = []
    log_lock = threading.Lock()
    monkeypatch.setattr(
        "ooc_optimizer.cfd.evaluate_cfd",
        _stub_evaluate_cfd_factory(call_log, log_lock, sleep_s=0.30),
    )

    runner = BORunner(cfg, pillar_config="none", H=200.0, topology="opposing")
    result = runner.run()

    # 1. Correct evaluation count (no races dropping/duplicating entries).
    assert len(runner.evaluations) == n_sobol + n_bo
    assert result["n_evaluations"] == n_sobol + n_bo

    # 2. Concurrency check: count maximum overlap of CFD calls in time.
    intervals = [(t, t + 0.30) for t, _ in call_log]
    starts = sorted(s for s, _ in intervals)
    ends = sorted(e for _, e in intervals)
    max_overlap, i, j, active = 0, 0, 0, 0
    while i < len(starts):
        if starts[i] < ends[j]:
            active += 1
            max_overlap = max(max_overlap, active)
            i += 1
        else:
            active -= 1
            j += 1
    assert max_overlap >= expected_min_concurrency, (
        f"batch_size={batch_size}: expected >={expected_min_concurrency} concurrent CFD calls, "
        f"observed peak overlap={max_overlap}"
    )

    # 3. Distinct worker threads observed iff batch_size > 1.
    distinct_threads = {name for _, name in call_log}
    if batch_size > 1:
        assert len(distinct_threads) >= 2
    else:
        assert len(distinct_threads) == 1


def test_q_batched_acquisition_returns_q_candidates(tmp_path, monkeypatch):
    """_optimize_acquisition(q=4) must return exactly 4 candidate tensors."""
    cfg = _make_config(tmp_path, batch_size=4, n_sobol=4, n_bo=0)
    monkeypatch.setattr(
        "ooc_optimizer.cfd.evaluate_cfd",
        _stub_evaluate_cfd_factory([], threading.Lock(), sleep_s=0.0),
    )
    runner = BORunner(cfg, pillar_config="none", H=200.0, topology="opposing")
    runner.run()  # populates train_X / train_Y / train_constraints
    runner._fit_gp()

    candidates = runner._optimize_acquisition(q=4)
    assert isinstance(candidates, list)
    assert len(candidates) == 4
    d = runner.train_X.shape[1]
    for c in candidates:
        assert c.shape == (d,)
