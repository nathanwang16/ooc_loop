"""
Unified ``tumor-chip`` CLI (Development Guide v2 §5.1).

Subcommands wrap the existing scripts so that after ``pip install
tumor-chip-design`` users have a single entry point::

    tumor-chip verify-scalar --template ooc_optimizer/cfd/template --output data/sv
    tumor-chip optimize      --config configs/default_config.yaml
    tumor-chip interpret     --results-dir data/results
    tumor-chip validate-3d   --config configs/default_config.yaml --bo-state ... --case-2d ...
    tumor-chip version

Each subcommand preserves the argparse API of the original script for
backward compatibility; the Typer wrapper exists so that ``tumor-chip
--help`` returns a single coherent usage instead of forcing users to
discover the scripts/ directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Typer is required for the tumor-chip CLI; install via pip install 'tumor-chip-design'"
    ) from exc


app = typer.Typer(
    name="tumor-chip",
    help="Inverse design + interpretability pipeline for tumor-on-chip chambers.",
    no_args_is_help=True,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"


def _run_script(script_name: str, argv: List[str]) -> int:
    path = _SCRIPTS / script_name
    if not path.exists():
        typer.echo(f"Script missing: {path}", err=True)
        raise typer.Exit(2)
    return subprocess.call([sys.executable, str(path), *argv])


@app.command("verify-scalar")
def verify_scalar(
    template: Path = typer.Option(_REPO_ROOT / "ooc_optimizer/cfd/template", "--template"),
    output: Path = typer.Option(_REPO_ROOT / "data/scalar_verification", "--output"),
    n_cells: int = typer.Option(100, "--n-cells"),
    L_mm: float = typer.Option(10.0, "--L-mm"),
    diffusivity: float = typer.Option(1e-10, "--diffusivity"),
    tol: float = typer.Option(0.02, "--tol"),
) -> None:
    """Run the Module 1.1 scalar-transport verification sweep (4 Pe values)."""
    argv = [
        "--template", str(template),
        "--output", str(output),
        "--n-cells", str(n_cells),
        "--L-mm", str(L_mm),
        "--diffusivity", str(diffusivity),
        "--tol", str(tol),
    ]
    raise typer.Exit(_run_script("run_scalar_verification.py", argv))


@app.command("optimize")
def optimize(
    config: Path = typer.Option(..., "--config"),
    parallel: bool = typer.Option(False, "--parallel"),
    single_target: bool = typer.Option(False, "--single-target"),
    override: Optional[Path] = typer.Option(None, "--override"),
    summary_out: Optional[Path] = typer.Option(None, "--summary-out"),
) -> None:
    """Run the Module 3.1 BO campaign (optionally multi-target)."""
    argv: List[str] = ["--config", str(config)]
    if parallel:
        argv.append("--parallel")
    if single_target:
        argv.append("--single-target")
    if override is not None:
        argv.extend(["--override", str(override)])
    if summary_out is not None:
        argv.extend(["--summary-out", str(summary_out)])
    raise typer.Exit(_run_script("run_optimization.py", argv))


@app.command("interpret")
def interpret(
    results_dir: Optional[Path] = typer.Option(None, "--results-dir"),
    state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
    sobol_n: int = typer.Option(1024, "--sobol-n"),
    loss_tol: float = typer.Option(0.1, "--loss-tol"),
) -> None:
    """Run the Module 3.3 interpretability analysis on one or more BO runs."""
    argv: List[str] = []
    if results_dir is None and state_dir is None:
        typer.echo("Either --results-dir or --state-dir is required.", err=True)
        raise typer.Exit(2)
    if results_dir is not None:
        argv.extend(["--results-dir", str(results_dir)])
    if state_dir is not None:
        argv.extend(["--state-dir", str(state_dir)])
    argv.extend(["--sobol-n", str(sobol_n), "--loss-tol", str(loss_tol)])
    raise typer.Exit(_run_script("run_interpretability.py", argv))


@app.command("validate-3d")
def validate_3d(
    config: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(..., "--output"),
    bo_state: Optional[Path] = typer.Option(None, "--bo-state"),
    orchestrator_summary: Optional[Path] = typer.Option(None, "--orchestrator-summary"),
    case_2d: Optional[Path] = typer.Option(None, "--case-2d"),
    target_profile: Optional[Path] = typer.Option(None, "--target-profile"),
    nz: int = typer.Option(25, "--nz"),
    z_grading: float = typer.Option(1.0, "--z-grading"),
) -> None:
    """Run the Module 4.1 3D CFD validation against a BO winner."""
    argv: List[str] = [
        "--config", str(config),
        "--output", str(output),
        "--nz", str(nz),
        "--z-grading", str(z_grading),
    ]
    if bo_state is not None:
        argv.extend(["--bo-state", str(bo_state)])
    if orchestrator_summary is not None:
        argv.extend(["--orchestrator-summary", str(orchestrator_summary)])
    if case_2d is not None:
        argv.extend(["--case-2d", str(case_2d)])
    if target_profile is not None:
        argv.extend(["--target-profile", str(target_profile)])
    raise typer.Exit(_run_script("run_3d_validation.py", argv))


@app.command("version")
def version() -> None:
    """Print the installed package version and exit."""
    from ooc_optimizer import __version__

    typer.echo(f"tumor-chip-design {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
