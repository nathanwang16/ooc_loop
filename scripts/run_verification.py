"""
Module 1.1 — Poiseuille flow verification script.

Runs a straight rectangular channel simulation and compares to the analytical
solution.  Includes mesh convergence study at 1×, 2×, 4× refinement.

Usage:
    python scripts/run_verification.py --config configs/default_config.yaml
    python scripts/run_verification.py --config configs/default_config.yaml --convergence
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ooc_optimizer.config import load_config
from ooc_optimizer.cfd.verification import (
    PoiseuilleSolution,
    extract_verification_results,
    run_mesh_convergence,
    run_openfoam_case,
    setup_verification_case,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_solution(config: dict) -> PoiseuilleSolution:
    """Construct the analytical solution from config fixed parameters."""
    fp = config["fixed_parameters"]
    return PoiseuilleSolution(
        L=fp["chamber_length_um"] * 1e-6,       # μm → m
        W=fp["inlet_width_um"] * 2 * 1e-6,      # verification channel: 2× inlet width = 1 mm
        H=config["discrete_levels"]["chamber_height"][0] * 1e-6,  # first height level
        Q_ul_min=config["baseline"]["Q"],
        mu=fp["fluid_viscosity_Pa_s"],
        rho=fp["fluid_density_kg_m3"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Module 1.1 — Poiseuille flow verification"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Run full mesh convergence study (1×, 2×, 4×)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/cases/verification"),
        help="Output directory for verification cases",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sol = _build_solution(config)

    logger.info("Verification parameters:")
    logger.info("  Channel: L=%.1f mm, W=%.1f mm, H=%.0f μm",
                sol.L * 1e3, sol.W * 1e3, sol.H * 1e6)
    logger.info("  Flow: Q=%.1f μL/min, U_mean=%.4e m/s, Re=%.2f",
                sol.Q_ul_min, sol.U_mean, sol.Re)
    logger.info("  Analytical: U_cl=%.4e m/s, τ_floor_mean=%.4f Pa",
                sol.U_centerline, sol.floor_wss_mean)
    logger.info("  Development length: %.3f mm (channel=%.0f mm)",
                sol.development_length * 1e3, sol.L * 1e3)

    template_dir = Path(config["paths"]["template_case"])

    if args.convergence:
        logger.info("Running mesh convergence study ...")
        results = run_mesh_convergence(
            template_dir=template_dir,
            output_dir=args.output,
            sol=sol,
        )
        for r in results:
            if r.get("converged"):
                logger.info(
                    "  %dx: %d cells, U_cl_err=%.4f%%, τ_mean_err=%.4f%%, pass=%s",
                    r["level"], r["n_cells"],
                    r.get("centerline_velocity_error", float("nan")) * 100,
                    r.get("tau_mean_error", float("nan")) * 100,
                    r.get("passed_2pct", False),
                )
            else:
                logger.warning("  %dx: FAILED to converge", r["level"])
    else:
        case_dir = args.output / "poiseuille_single"
        nx, ny = 200, 20
        logger.info("Setting up single verification case (nx=%d, ny=%d) ...", nx, ny)

        setup_verification_case(
            case_dir=case_dir,
            template_dir=template_dir,
            sol=sol,
            nx=nx,
            ny=ny,
        )

        converged = run_openfoam_case(case_dir)
        if not converged:
            logger.error("Simulation did not converge. Check logs in %s", case_dir / "logs")
            sys.exit(1)

        results = extract_verification_results(case_dir, sol)

        results_path = case_dir / "verification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to %s", results_path)
        if results["passed_2pct"]:
            logger.info("VERIFICATION PASSED (<2%% error)")
        else:
            logger.warning("VERIFICATION FAILED (>2%% error)")
            sys.exit(1)


if __name__ == "__main__":
    main()
