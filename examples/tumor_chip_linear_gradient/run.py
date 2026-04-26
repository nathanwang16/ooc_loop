"""
Example — inverse design for a linear concentration gradient.

Sweeps all three inlet topologies (opposing, same_side_Y, asymmetric_lumen)
at fixed pillar="none" and H=200 μm, with 120 evals per topology (24 Sobol
+ 96 BO).  After BO finishes, per-topology interpretability summaries are
written next to each ``bo_<topology>_…/`` directory and a
``comparison_report.md`` is generated comparing best L², feasibility,
dominant parameters and tightest tolerances across the three topologies.

Usage:
    python examples/tumor_chip_linear_gradient/run.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ooc_optimizer.config import load_config
from ooc_optimizer.interpretability import analyse_winner
from ooc_optimizer.optimization.orchestrator import run_all_configurations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tumor_chip_linear_gradient.run")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _merge_overrides(base: dict, override: dict) -> dict:
    for key, value in override.items():
        base[key] = value
    return base


def _per_run_interpretability(all_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run Module 3.3 on every BO state directory, not just the global winner."""
    summaries: List[Dict[str, Any]] = []
    for run in all_runs:
        state_dir = Path(run["state_dir"])
        if run.get("best_feasible") is None:
            logger.warning(
                "Skipping interpretability for %s — no feasible BO point found",
                run.get("config_name"),
            )
            summaries.append({
                "config_name": run.get("config_name"),
                "topology": run.get("topology"),
                "state_dir": str(state_dir),
                "best_L2": None,
                "skipped_reason": "no feasible point",
            })
            continue
        try:
            summary = analyse_winner(
                state_dir,
                sobol_n_samples=512,
                tolerance_loss_tolerance=0.1,
            )
        except Exception as exc:
            logger.exception("Interpretability failed for %s: %s", state_dir, exc)
            summaries.append({
                "config_name": run.get("config_name"),
                "topology": run.get("topology"),
                "state_dir": str(state_dir),
                "best_L2": run["best_feasible"]["L2_to_target"],
                "skipped_reason": f"exception: {exc}",
            })
            continue
        summary["best_L2"] = run["best_feasible"]["L2_to_target"]
        summary["best_params"] = run["best_feasible"]["params"]
        summary["n_evaluations"] = run.get("n_evaluations")
        summary["n_feasible"] = sum(1 for e in run.get("evaluations", []) if e.get("feasible"))
        summary["config_name"] = run.get("config_name")
        summaries.append(summary)
    return summaries


def _format_pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _write_comparison_report(
    summaries: List[Dict[str, Any]], output_path: Path, winner_config_name: str
) -> Path:
    """Write a markdown comparison across all topologies that produced a winner."""
    lines: List[str] = [
        "# Cross-topology comparison",
        "",
        "Bayesian-optimization sweep of three inlet topologies for a linear",
        "x-gradient target.  120 evaluations per topology (24 Sobol + 96 BO);",
        "pillar_config=`none`, H=200 μm fixed.",
        "",
        f"**Global winner**: `{winner_config_name}`",
        "",
        "## Performance summary",
        "",
        "| Topology | Best L² | Feasible | n_evals | Active dim |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        if s.get("skipped_reason"):
            lines.append(
                f"| `{s['topology']}` | — | — | {s.get('n_evaluations', '—')} | "
                f"_{s['skipped_reason']}_ |"
            )
            continue
        n_feas = s.get("n_feasible", 0)
        n_evals = s.get("n_evaluations", 0)
        active_dim = len(s["sobol"]["names"])
        lines.append(
            f"| `{s['topology']}` | {s['best_L2']:.4f} | {_format_pct(n_feas, n_evals)} | "
            f"{n_evals} | {active_dim} |"
        )

    lines.extend([
        "",
        "## Dominant parameters per topology (Sobol S_T ≥ 20%·max)",
        "",
        "| Topology | Dominant | Inert (S_T < 5%·max) |",
        "|---|---|---|",
    ])
    for s in summaries:
        if s.get("skipped_reason"):
            continue
        names = s["sobol"]["names"]
        ST = s["sobol"]["ST"]
        st_max = max(ST) if ST else 1.0
        dominant = [n for n, st in zip(names, ST) if st >= 0.2 * st_max]
        inert = [n for n, st in zip(names, ST) if st < 0.05 * st_max]
        dom_str = ", ".join(f"`{n}`" for n in dominant) or "—"
        inert_str = ", ".join(f"`{n}`" for n in inert) or "—"
        lines.append(f"| `{s['topology']}` | {dom_str} | {inert_str} |")

    lines.extend([
        "",
        "## Sobol sanity gate (ΣS_1, ΣS_T)",
        "",
        "ΣS_1 ≤ 1 always.  ΣS_T ≤ 1 only if interactions are absent;",
        "ΣS_T » 1 (≳ 1.5) suggests the GP is overfitting and the indices are",
        "not yet trustworthy.",
        "",
        "| Topology | ΣS_1 | ΣS_T | Trustworthy? |",
        "|---|---|---|---|",
    ])
    for s in summaries:
        if s.get("skipped_reason"):
            continue
        sum_s1 = sum(s["sobol"]["S1"])
        sum_st = sum(s["sobol"]["ST"])
        verdict = "ok" if sum_st < 1.5 else "**suspect**"
        lines.append(f"| `{s['topology']}` | {sum_s1:.2f} | {sum_st:.2f} | {verdict} |")

    lines.extend([
        "",
        "## Tightest fabrication tolerances across topologies (sum of ±Δ_norm)",
        "",
        "Lower = tighter (parameter must be controlled more carefully).",
        "",
        "| Topology | Tightest 3 (param: ±Δ_norm) |",
        "|---|---|",
    ])
    for s in summaries:
        if s.get("skipped_reason"):
            continue
        intervals = sorted(
            s["tolerance_intervals"],
            key=lambda iv: iv["delta_plus_norm"] + iv["delta_minus_norm"],
        )[:3]
        cells = ", ".join(
            f"`{iv['name']}`: ±{(iv['delta_plus_norm']+iv['delta_minus_norm'])/2:.3f}"
            for iv in intervals
        )
        lines.append(f"| `{s['topology']}` | {cells} |")

    lines.extend([
        "",
        "## Boundary-pinning check",
        "",
        "Parameters whose normalised optimum is within 3% of an active bound",
        "indicate the search box, not the physics, is binding.  Widen those",
        "bounds and re-run if the goal is a true global optimum.",
        "",
        "| Topology | Pinned at bounds |",
        "|---|---|",
    ])
    for s in summaries:
        if s.get("skipped_reason"):
            continue
        order = s["sobol"]["names"]
        x_full = s["x_star_norm"]
        # x_star_norm has full PARAMETER_ORDER length; map active names to it.
        active_names = s["sobol"]["names"]
        # Recover positions: the same x_star_norm slots that match active_names.
        # We rely on the ordering convention from PARAMETER_ORDER →
        # the interpretability summary stores names in active-only order, but
        # x_star_norm is full-length.  We'll just check the active dims by
        # matching active_names back to PARAMETER_ORDER.
        from ooc_optimizer.optimization.bo_loop import PARAMETER_ORDER
        pinned = []
        for n in active_names:
            idx = PARAMETER_ORDER.index(n)
            v = x_full[idx]
            if v <= 0.03:
                pinned.append(f"`{n}`(↓)")
            elif v >= 0.97:
                pinned.append(f"`{n}`(↑)")
        lines.append(f"| `{s['topology']}` | {', '.join(pinned) if pinned else '—'} |")

    lines.extend([
        "",
        "## Per-topology design notes",
        "",
    ])
    for s in summaries:
        if s.get("skipped_reason"):
            continue
        h_md = s.get("heuristic_markdown")
        if h_md:
            rel = Path(h_md).resolve()
            lines.append(f"- `{s['topology']}` → {rel}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    config = load_config(REPO / "configs" / "default_config.yaml")
    with open(HERE / "config.yaml", "r") as f:
        overrides = yaml.safe_load(f) or {}
    config = _merge_overrides(config, overrides)

    topologies = config.get("discrete_levels", {}).get("inlet_topology", [])
    logger.info("Sweeping topologies: %s", topologies)
    results = run_all_configurations(config, parallel=False)
    winner = results.get("winner")
    all_runs = results.get("all_runs", [])

    if winner is None:
        logger.error("No feasible winner across any topology; interpretability skipped.")
    else:
        print(
            f"Global winner: topology={winner['topology']}, pillar={winner['pillar_config']}, "
            f"H={winner['H']} μm, L2={winner['L2_to_target']:.4f}"
        )

    summaries = _per_run_interpretability(all_runs)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    winner_name = winner["config_name"] if winner else "(none)"
    report_path = _write_comparison_report(
        summaries, results_dir / "comparison_report.md", winner_name
    )
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    main()
