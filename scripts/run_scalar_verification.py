"""Module 1.1 — 1D advection-diffusion verification driver (v2).

Runs scalarTransportFoam on a 1D channel with a uniform velocity at four
Peclet numbers spanning the design-relevant regime, and compares results to
the analytic solution ``C(x) = (1 - exp(-Pe*(1-x/L))) / (1 - exp(-Pe))``.

Pass criterion: L2 relative error < 2% on a 100-cell mesh.  Results (pass /
fail per Pe, errors, case paths) are written to
``data/scalar_verification/scalar_verification_results.json`` and echoed to
stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ooc_optimizer.cfd.scalar import (
    DEFAULT_DIFFUSIVITY_M2_S,
    run_scalar_verification_1d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scalar_verification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("ooc_optimizer/cfd/template"))
    parser.add_argument("--output", type=Path, default=Path("data/scalar_verification"))
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--L-mm", type=float, default=10.0, help="channel length [mm]")
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=DEFAULT_DIFFUSIVITY_M2_S,
        help="base diffusivity [m^2/s]",
    )
    parser.add_argument(
        "--pe",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
        help="Peclet numbers to sweep",
    )
    parser.add_argument("--tol", type=float, default=0.02, help="L2 error pass threshold")
    args = parser.parse_args()

    results = run_scalar_verification_1d(
        template_dir=args.template,
        output_dir=args.output,
        L_m=args.L_mm * 1e-3,
        n_cells=args.n_cells,
        pe_values=tuple(args.pe),
        diffusivity_m2_s=args.diffusivity,
        tolerance_l2=args.tol,
    )

    print("\n=== Scalar Verification Summary ===")
    print(json.dumps(results, indent=2))
    if all(r.get("passed") for r in results):
        print("\nALL PASS")
    else:
        print("\nAT LEAST ONE FAILURE — see JSON above")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
