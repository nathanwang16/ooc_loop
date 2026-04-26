"""
Diagnostic — single CFD evaluation at the default-config baseline parameters
against the linear-gradient target.  Used to test whether the BO winner
(L2=0.6343) beats a "do-nothing" baseline, which decides between hypothesis
"design-space ceiling" and "BO stall" (see plans/proceed-with-all-the-tingly-map.md).

No optimisation, no defaults: every value comes from configs/default_config.yaml's
`baseline:` block.  Topology fixed to `opposing` to match the BO winner.

Pass `--axis y` to evaluate against a transverse linear gradient instead — this
tests whether the topology's natural mixing axis (perpendicular to coflowing
inlets) is the only direction in which a linear gradient is physically achievable.

Outputs:
    examples/tumor_chip_linear_gradient/data/diagnostic/baseline_metrics_<axis>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ooc_optimizer.config import load_config
from ooc_optimizer.optimization.bo_loop import BORunner
from ooc_optimizer.optimization.objectives import build_target_profile
from ooc_optimizer.cfd.solver import evaluate_cfd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("diagnostic_baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("x", "y"), default="x",
                        help="Target axis (default: x, matching BO campaign).")
    parser.add_argument("--target-kind", choices=("linear_gradient", "step"), default="linear_gradient",
                        help="Target shape (default: linear_gradient).")
    parser.add_argument("--sharpness-frac", type=float, default=0.05,
                        help="For --target-kind step: transition-region width as fraction of axis (default 0.05).")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "configs" / "default_config.yaml")
    if args.target_kind == "linear_gradient":
        cfg["target_profile"] = {
            "kind": "linear_gradient", "axis": args.axis, "c_high": 1.0, "c_low": 0.0,
        }
    else:
        cfg["target_profile"] = {
            "kind": "step", "step_axis": args.axis, "step_frac": 0.5,
            "sharpness_frac": args.sharpness_frac, "c_high": 1.0, "c_low": 0.0,
        }
    target_tag = f"{args.target_kind}_{args.axis}"
    if args.target_kind == "step":
        target_tag += f"_sharp{args.sharpness_frac:.2f}".replace(".", "p")

    if "baseline" not in cfg:
        raise KeyError("configs/default_config.yaml must define `baseline:` block")
    b = cfg["baseline"]

    required = ("W", "d_p", "s_p", "theta", "Q_total", "r_flow", "delta_W", "pillar_config", "H")
    missing = [k for k in required if k not in b]
    if missing:
        raise KeyError(f"baseline: missing keys {missing}")

    params = {
        "W": float(b["W"]),
        "d_p": float(b["d_p"]),
        "s_p": float(b["s_p"]),
        "theta": float(b["theta"]),
        "Q_total": float(b["Q_total"]),
        "r_flow": float(b["r_flow"]),
        "delta_W": float(b["delta_W"]),
    }
    pillar_config = str(b["pillar_config"])
    H_um = float(b["H"])

    out_root = REPO_ROOT / "examples" / "tumor_chip_linear_gradient" / "data" / "diagnostic"
    stl_dir = out_root / "stl"
    cases_dir = out_root / "cases"
    stl_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["stl_output_dir"] = str(stl_dir)
    cfg["paths"]["case_output_dir"] = str(cases_dir)

    target = build_target_profile(cfg["target_profile"])
    cfd_cfg = BORunner._build_cfd_config(cfg)

    log.info("Baseline params: %s", params)
    log.info("Topology=opposing  pillar=%s  H=%.0f um  target=%s", pillar_config, H_um, cfg["target_profile"])

    metrics = evaluate_cfd(
        params=params,
        pillar_config=pillar_config,
        H_um=H_um,
        config=cfd_cfg,
        topology="opposing",
        target_profile=target,
    )

    out_json = out_root / f"baseline_metrics_{target_tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2, default=str))

    print()
    print("=== baseline diagnostic vs linear-gradient target ===")
    for k in ("L2_to_target", "C_mean", "C_std", "monotonicity", "grad_sharpness",
              "tau_mean", "f_dead", "converged_U", "converged_C", "mesh_ok"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k:<18} = {v}")
    print(f"\nFull metrics dict: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
