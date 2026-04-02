"""
File I/O and path management utilities.

Handles OpenFOAM case directory setup, STL path resolution, and
safe directory creation for evaluation outputs.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_case_directory(template_dir: Path, case_dir: Path) -> Path:
    """Copy the OpenFOAM template case to a new case directory.

    Parameters
    ----------
    template_dir : Path
        Path to the template case (system/, constant/, 0/).
    case_dir : Path
        Destination path for the new case.

    Returns
    -------
    case_dir : Path

    Raises
    ------
    FileNotFoundError
        If template_dir does not exist.
    FileExistsError
        If case_dir already exists.
    """
    template_dir = Path(template_dir)
    case_dir = Path(case_dir)

    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")
    if case_dir.exists():
        raise FileExistsError(f"Case directory already exists: {case_dir}")

    shutil.copytree(template_dir, case_dir)
    logger.info("Created case directory: %s", case_dir)
    return case_dir


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_stl_path(output_dir: Path, name: str) -> Path:
    """Build the full path for an STL file within the output directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    return output_dir / f"{name}.stl"
