"""
OpenFOAM ASCII field file parser.

Reads internalField data from OpenFOAM volScalarField and volVectorField
files into numpy arrays.  Supports both 'uniform' and 'nonuniform List<...>'
formats.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_RE_NONUNIFORM_HEADER = re.compile(
    r"internalField\s+nonuniform\s+List<(\w+)>\s*\n\s*(\d+)\s*\n\s*\("
)
_RE_UNIFORM_SCALAR = re.compile(
    r"internalField\s+uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;"
)
_RE_UNIFORM_VECTOR = re.compile(
    r"internalField\s+uniform\s+\(\s*([-+\d.eE\s]+)\s*\)\s*;"
)


def read_scalar_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volScalarField file and return values as 1-D array."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Field file not found: {filepath}")

    text = filepath.read_text()

    m_uniform = _RE_UNIFORM_SCALAR.search(text)
    if m_uniform:
        raise ValueError(
            f"Field {filepath.name} is uniform ({m_uniform.group(1)}); "
            "cannot convert to per-cell array without cell count"
        )

    m = _RE_NONUNIFORM_HEADER.search(text)
    if m is None:
        raise ValueError(f"Cannot parse internalField in {filepath}")

    field_type = m.group(1)
    n_cells = int(m.group(2))

    if field_type != "scalar":
        raise ValueError(f"Expected scalar field, got '{field_type}' in {filepath}")

    data_start = m.end()
    data_section = text[data_start:]
    closing = data_section.index(")")
    raw = data_section[:closing]

    values = np.fromstring(raw, sep="\n", count=n_cells)
    if values.size != n_cells:
        raise ValueError(
            f"Expected {n_cells} values in {filepath}, parsed {values.size}"
        )

    logger.debug("Read %d scalar values from %s", n_cells, filepath)
    return values


def read_vector_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volVectorField file and return Nx3 array."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Field file not found: {filepath}")

    text = filepath.read_text()

    m_uniform = _RE_UNIFORM_VECTOR.search(text)
    if m_uniform:
        raise ValueError(
            f"Field {filepath.name} is uniform; "
            "cannot convert to per-cell array without cell count"
        )

    m = _RE_NONUNIFORM_HEADER.search(text)
    if m is None:
        raise ValueError(f"Cannot parse internalField in {filepath}")

    field_type = m.group(1)
    n_cells = int(m.group(2))

    if field_type != "vector":
        raise ValueError(f"Expected vector field, got '{field_type}' in {filepath}")

    data_start = m.end()
    data_section = text[data_start:]
    closing_paren_depth = 0
    end_idx = 0
    for i, ch in enumerate(data_section):
        if ch == "(":
            closing_paren_depth += 1
        elif ch == ")":
            if closing_paren_depth == 0:
                end_idx = i
                break
            closing_paren_depth -= 1

    raw = data_section[:end_idx]

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if len(numbers) != n_cells * 3:
        raise ValueError(
            f"Expected {n_cells * 3} components in {filepath}, found {len(numbers)}"
        )

    values = np.array([float(x) for x in numbers]).reshape(n_cells, 3)
    logger.debug("Read %d vector values from %s", n_cells, filepath)
    return values


def find_latest_time(case_dir: Path) -> Optional[Path]:
    """Find the latest numerical time directory in an OpenFOAM case.

    Skips directory '0' (initial conditions) and 'constant'.
    Returns None if no result time directories exist.
    """
    case_dir = Path(case_dir)
    time_dirs = []
    for d in case_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
            if t > 0:
                time_dirs.append((t, d))
        except ValueError:
            continue

    if not time_dirs:
        return None

    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]


def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Read cell centres written by 'postProcess -func writeCellCentres'.

    Returns Nx3 array of (x, y, z) coordinates in meters.
    Looks for the C (volVectorField) file in the time '0' or latest time directory.
    """
    case_dir = Path(case_dir)

    for search_dir in [case_dir / "0", find_latest_time(case_dir)]:
        if search_dir is None:
            continue
        c_file = search_dir / "C"
        if c_file.exists():
            return read_vector_field(c_file)

    for search_dir in [case_dir / "0", find_latest_time(case_dir)]:
        if search_dir is None:
            continue
        ccx = search_dir / "ccx"
        ccy = search_dir / "ccy"
        ccz = search_dir / "ccz"
        if ccx.exists() and ccy.exists() and ccz.exists():
            x = read_scalar_field(ccx)
            y = read_scalar_field(ccy)
            z = read_scalar_field(ccz)
            return np.column_stack([x, y, z])

    raise FileNotFoundError(
        f"Cell centre data not found in {case_dir}. "
        "Run 'postProcess -func writeCellCentres' first."
    )
