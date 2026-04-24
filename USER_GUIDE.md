# User Guide — tumor-chip-design

A step-by-step, command-by-command walkthrough for driving the pipeline end
to end.  The guide is written for a biologist / mechanical engineer who is
comfortable on the command line but has not used OpenFOAM or BoTorch before.
No part of this guide asks you to read source code; every science question
you need to answer about the pipeline is addressed where the relevant
command first appears.

If you only have 30 minutes, skip to **[§2](#2-first-thirty-minutes-verify-the-install)**
and run through it once.  After that, you will have a working install and a
clear mental model of what each stage produces.

---

## Contents

1. [What this software does, in one paragraph](#1-what-this-software-does-in-one-paragraph)
2. [First thirty minutes — verify the install](#2-first-thirty-minutes-verify-the-install)
3. [A five-minute mental model of the pipeline](#3-a-five-minute-mental-model-of-the-pipeline)
4. [Smoke tests — confirm each stage works before you trust it](#4-smoke-tests-confirm-each-stage-works-before-you-trust-it)
5. [Your first real BO run (the 15-minute example)](#5-your-first-real-bo-run-the-15-minute-example)
6. [Reading what came out](#6-reading-what-came-out)
7. [The headline: interpretability — dominant parameters and tolerances](#7-the-headline-interpretability--dominant-parameters-and-tolerances)
8. [3D validation — did the 2D shortcut hold?](#8-3d-validation--did-the-2d-shortcut-hold)
9. [Running the full campaign (weekend-scale)](#9-running-the-full-campaign-weekend-scale)
10. [Going to the bench — what changes in fabrication and imaging](#10-going-to-the-bench--what-changes-in-fabrication-and-imaging)
11. [Troubleshooting — where to look when something fails](#11-troubleshooting--where-to-look-when-something-fails)
12. [Glossary of files and folders](#12-glossary-of-files-and-folders)

---

## 1. What this software does, in one paragraph

You give it a **target concentration profile** (e.g. "a smooth gradient
from `C = 1` at the drug inlet to `C = 0` at the medium inlet across the
chamber"); it searches a 7-dimensional parametric chip-geometry space
using ~60 CFD simulations per discrete configuration, returns the geometry
whose achieved concentration field best matches the target, and — most
importantly — tells you **which of the geometry's dimensions actually
matter and how tightly you need to hold each one during fabrication**.
That last sentence is the real deliverable.  The BO is the means; the
interpretability output is the end.

---

## 2. First thirty minutes — verify the install

### 2.1 What you need on your laptop / workstation

- macOS (Apple Silicon or Intel) or Linux.  Windows is not supported;
  OpenFOAM only runs on Unix-family systems.
- Python 3.10, 3.11, or 3.12.  `3.11` is the recommended default.
- A working **OpenFOAM v2406** install.  On macOS:
  ```bash
  brew tap gerlero/openfoam
  brew install gerlero/openfoam/openfoam@2406
  ```
  On Linux use the OpenCFD apt packages or the Docker image (see §2.4).
- `git` and roughly 10 GB of free disk space for the simulation cases.

### 2.2 Set up a Python environment

Use conda because the convention across the repo assumes an `ooc`
environment:

```bash
conda create -n ooc python=3.11 -y
conda activate ooc
```

Then install the package in editable mode so you can tinker without
re-installing every time:

```bash
cd ~/code                   # or wherever you keep repos
git clone <repo-url> tumor-chip-design
cd tumor-chip-design
pip install -e ".[dev,test,docs]"
pre-commit install          # optional but recommended if you'll edit code
```

### 2.3 Confirm the install

```bash
tumor-chip version                    # should print "tumor-chip-design 0.6.0"
pytest -q --deselect tests/test_verification.py::TestOpenFOAMIntegration \
          --deselect tests/test_cfd.py \
          --deselect tests/test_geometry.py::TestGeometryValidation \
          --deselect tests/test_geometry.py::TestGeometryGeneration \
          --deselect tests/test_meshing.py
```

You should see **101 passed**.  The five `--deselect` flags skip a handful
of v1 test stubs that were never implemented and a live-OpenFOAM test that
requires a mesh convergence study; neither matters for the pipeline.

### 2.4 Alternative: Docker (no local OpenFOAM needed)

If you do not want to install OpenFOAM on the host, use the bundled Docker
image.  It layers Python 3.11 and the package on top of
`opencfd/openfoam-default:2406`:

```bash
docker compose -f docker/docker-compose.yml run --rm tumor-chip-shell
# inside the container:
tumor-chip version
```

All output (cases, logs, figures) lands under `./data/` on the host and
survives container restarts.

### What you should see if everything is fine

- `tumor-chip version` prints a version string, not a traceback.
- `pytest` reports `101 passed`.
- `which simpleFoam` (or `which openfoam2406`) returns a real path.

If any of those fail, jump to **[§11 Troubleshooting](#11-troubleshooting--where-to-look-when-something-fails)**.

---

## 3. A five-minute mental model of the pipeline

The pipeline is a linear chain with one optimisation loop wrapping four
of the six stages:

```
  ┌─────────────────────────────────────────────────┐
  │                  BO loop (Module 3.1)           │
  │  1. pick candidate parameters x                 │
  │                       │                         │
  │                       ▼                         │
  │  2. geometry (Module 1.2) ──► blockMeshDict     │
  │                       │                         │
  │                       ▼                         │
  │  3. blockMesh  (Module 2.1)                     │
  │                       │                         │
  │                       ▼                         │
  │  4. simpleFoam (Module 2.2 — momentum)          │
  │                       │                         │
  │                       ▼                         │
  │  5. scalarTransportFoam (Module 2.2 — tracer)   │
  │                       │                         │
  │                       ▼                         │
  │  6. metrics — L²(C, C_target), τ, f_dead, …     │
  │                       │                         │
  └───────────────────────┼─────────────────────────┘
                          ▼
  7. winner → Module 3.3 interpretability
                          │
                          ▼
  8. winner → Module 4.1 3D validation
                          │
                          ▼
  9. winner → §10 fabricate + image
```

A few terms worth learning before you run anything:

- **Topology** — where the two inlets enter the chamber.  Three choices:
  `opposing`, `same_side_Y`, `asymmetric_lumen`.  (See
  `docs/concepts/parametric-geometry.md` for pictures.)
- **Discrete configuration** — one tuple of
  `(pillar_config, chamber_height, inlet_topology)`.  The default config
  has `4 × 2 × 3 = 24` of them; each gets its own BO.
- **Target profile** — the concentration field you want to match.  Three
  shipping kinds: `linear_gradient`, `bimodal`, `step` (see
  `docs/concepts/inverse-design.md`).
- **`L2_to_target`** — the primary objective; lower is better.  `0` means
  the achieved field matches the target exactly.
- **`τ_mean` and `f_dead`** — hard constraints: the flow must be neither
  pathologically slow nor pathologically fast, and the chamber must not
  contain > 5% dead-zone area.  These are filters, not things the BO
  optimises.
- **Winner** — the lowest-`L2`, constraint-satisfying geometry across a
  BO run (or across all 24 BO runs, when you sweep all configurations).

---

## 4. Smoke tests — confirm each stage works before you trust it

### 4.1 Verify the scalar solver (2–3 minutes)

Before the BO can trust `scalarTransportFoam`, we check it against an
analytic solution at four Péclet numbers:

```bash
tumor-chip verify-scalar \
    --output data/scalar_verification \
    --n-cells 100
```

**What this runs.** Four 1-D channel cases (Pe ∈ {1, 10, 100, 1000}) on
a 100-cell mesh; the Python layer compares the simulated `C(x)` to the
textbook solution
`C(ξ) = [1 − exp(−Pe·(1−ξ))] / [1 − exp(−Pe)]`.

**What you should see.** Console ends with:

```
=== Scalar Verification Summary ===
[ {... "Pe": 1.0,    "L2_rel_error": ~0.001, "passed": true, ...},
  {... "Pe": 10.0,   "L2_rel_error": ~0.002, "passed": true, ...},
  {... "Pe": 100.0,  "L2_rel_error": ~0.01,  "passed": true, ...},
  {... "Pe": 1000.0, "L2_rel_error": ~0.02,  "passed": true, ...} ]

ALL PASS
```

A pass means all four Pe values hit `L2_rel_error < 2 %` vs the analytic
solution.  Fail → do not proceed; read the generated log at
`data/scalar_verification/ad_1d_Pe1/scalarTransportFoam.log`.

### 4.2 Verify the momentum solver (5–8 minutes)

The v1 Poiseuille check is retained and is the cleanest way to prove that
`simpleFoam` is producing trustworthy floor velocities:

```bash
python scripts/run_verification.py
```

**What you should see.** A Poiseuille case runs at three mesh refinement
levels; each reports a centerline-velocity error and floor-WSS error
below 2 %.  Results land in `data/verification/convergence_results.json`.

### 4.3 Confirm the geometry layer builds cleanly

The geometry layer is pure Python and never touches OpenFOAM, so it
cannot *fail* silently — but it can produce invalid parameters.  Run:

```bash
pytest -q tests/test_topology_blockmesh.py tests/test_topology_blockmesh_3d.py
```

Both should report **all green** in under 5 seconds.  This proves the
2D and 3D blockMesh generators emit the required patches
(`inlet_drug`, `inlet_medium`, `outlet`, `walls`, `floor`, `ceiling`)
for every topology.

---

## 5. Your first real BO run (the 15-minute example)

Time to do actual science.

### 5.1 Run the shrunken linear-gradient example

```bash
python examples/tumor_chip_linear_gradient/run.py
```

**What this runs.** One discrete configuration
(`opposing` topology × `no pillars` × `H = 200 μm`) with 20 evaluations
(8 Sobol + 12 BO).  End-to-end time on 8 cores: ~15 minutes.

**What you should see.** The console prints one line per BO iteration,
e.g.:

```
2026-04-22 09:30:10 INFO BO iter 1/12 (opposing, none, H=200): L2=0.4123
2026-04-22 09:30:58 INFO BO iter 2/12 (opposing, none, H=200): L2=0.3844
...
2026-04-22 09:46:02 INFO BO iter 12/12 (opposing, none, H=200): L2=0.1237

Winner: topology=opposing, pillar=none, H=200.0 μm, L2=0.1237
Design heuristics: examples/tumor_chip_linear_gradient/data/results/bo_opposing_none_H200/interpretability/design_heuristics.md
```

**Why `L2` should be decreasing.** The BO should drive `L2_to_target`
monotonically downward (with occasional blips — exploration, not
exploitation).  A final `L2` near or below `0.1` means the achieved field
matches the target to within ~10 % RMS; anything above `0.3` after 20
evals suggests the constraints are cutting off the design space and you
should loosen `τ_mean_min` in the config.

### 5.2 Where did everything go?

```
examples/tumor_chip_linear_gradient/data/
├── cases/                                        # per-evaluation OpenFOAM runs
│   └── run_opposing_none_H200_<timestamp>/
│       ├── blockMesh.log
│       ├── simpleFoam.log
│       ├── scalarTransportFoam.log
│       └── <latest_time>/{U, p, T, wallShearStress}
├── results/
│   ├── bo_opposing_none_H200/                    # BO state + interpretability
│   │   ├── evaluations.json                      # full trace (params, metrics)
│   │   ├── gp_model_state.pt                     # trained GP weights
│   │   └── interpretability/
│   │       ├── sobol.png                         # ← global sensitivity bar chart
│   │       ├── local_sensitivity.png             # ← |∂μ/∂x| ranking at optimum
│   │       ├── tolerance.png                     # ← per-parameter ± tolerance
│   │       ├── summary.json
│   │       └── design_heuristics.md              # ← the plain-English deliverable
│   └── evaluations_opposing_none_H200.jsonl      # per-eval append-only log
├── figures/                                      # (empty until you run the plotter)
└── stl/                                          # optional STL exports
```

---

## 6. Reading what came out

### 6.1 The per-evaluation log

```bash
head -n 1 examples/tumor_chip_linear_gradient/data/results/evaluations_opposing_none_H200.jsonl | python -m json.tool
```

shows one evaluation record.  The fields you care about are:

- `params` — the 7-D parameter vector that was evaluated.
- `metrics.L2_to_target` — the primary objective (lower = better match).
- `metrics.tau_mean`, `metrics.f_dead` — the constraints.
- `metrics.converged` — `true` iff *both* `simpleFoam` and
  `scalarTransportFoam` hit their residual targets.
- `metrics.case_dir` — path to the OpenFOAM case if you want to look at
  the velocity / concentration fields in ParaView.

### 6.2 Plot the achieved concentration field

```bash
python - <<'PY'
from pathlib import Path
from ooc_optimizer.analysis import plot_concentration_contour, plot_residual_field
from ooc_optimizer.optimization.objectives import linear_gradient
import json

run = "examples/tumor_chip_linear_gradient/data/results/bo_opposing_none_H200"
evals = json.load(open(f"{run}/evaluations.json"))
best_idx = min(
    (i for i, r in enumerate(evals["evaluations"]) if r["feasible"]),
    key=lambda i: evals["evaluations"][i]["objective"],
)
case = Path(evals["evaluations"][best_idx]["metrics"]["case_dir"])
target = linear_gradient(axis="x", c_high=1.0, c_low=0.0)

plot_concentration_contour(case, L=10e-3, W=evals["evaluations"][best_idx]["params"]["W"]*1e-6,
                           output_path=f"{run}/figures/contour.png")
plot_residual_field(case, target, L=10e-3,
                    W=evals["evaluations"][best_idx]["params"]["W"]*1e-6,
                    output_path=f"{run}/figures/residual.png")
print("Wrote contour.png and residual.png under", run)
PY
```

**What you should see.**
- `contour.png` — a smooth blue→yellow gradient across the chamber if the
  winner matched a linear gradient.  Bands parallel to `y` mean the flow
  was dominated by advection in `x` — expected for `Pe ≫ 1`.
- `residual.png` — mostly near-zero (pale), with the biggest deviations
  near the inlet walls where diffusion has not yet smoothed the corner
  effects.  If the residual is > 0.3 over more than ~10 % of the floor,
  your BO did not actually find a good match — re-run with more
  iterations.

### 6.3 Reproducing the centerline and streamline plots

Same idea, different plot functions (already imported in
`ooc_optimizer.analysis.__init__`):

```python
from ooc_optimizer.analysis import plot_centerline_profile, plot_streamline_overlay
```

Use when you want to show a reviewer "the `C(x, W/2)` overlay matches the
target" or "the flow goes where you'd expect it to go".

---

## 7. The headline: interpretability — dominant parameters and tolerances

This is the part of the software that you should be showing to
collaborators.  The BO gave you *a* good geometry; the interpretability
layer tells you *why* and *how precisely you need to fabricate it*.

### 7.1 Run it

The `examples/tumor_chip_linear_gradient/run.py` script already triggers
`analyse_winner` at the end.  If you want to re-run it standalone:

```bash
tumor-chip interpret \
    --state-dir examples/tumor_chip_linear_gradient/data/results/bo_opposing_none_H200 \
    --sobol-n 512 \
    --loss-tol 0.10
```

**What the flags mean.**
- `--sobol-n 512` — number of Saltelli samples used for Sobol indices.
  `1024` is the default for the full campaign; `512` is fine for a
  quickstart.  Larger = tighter confidence intervals, ~linearly longer
  runtime.
- `--loss-tol 0.10` — how much L² degradation defines the "edge" of each
  parameter's tolerance interval.  `0.10` = "how far can I move this
  parameter before the field match degrades by 10 %?"

### 7.2 What to look at and what it means

Inside `.../interpretability/` you get four artefacts:

| File                        | What it shows                                            | How to read it |
|-----------------------------|----------------------------------------------------------|----------------|
| `sobol.png`                 | Bar chart of first-order (`S₁`) and total-order (`Sₜ`) indices per parameter | Tall `Sₜ` bar = **this parameter dominates the field**. Parameters with `Sₜ < 0.05·max(Sₜ)` can be held loosely. |
| `local_sensitivity.png`     | Horizontal bar chart of `|∂μ/∂x|` at the BO optimum       | Complements the Sobol chart — *at* the winning point, which parameter changes the outcome most per unit move? |
| `tolerance.png`             | Asymmetric error bars showing `± Δx` in **physical units** | Narrow error bar = **tight tolerance** (fabricate carefully). Wide error bar = loose tolerance. |
| `design_heuristics.md`      | Auto-generated English summary                           | This is the page you hand to a colleague who doesn't know the pipeline. |

**Example `design_heuristics.md` for a linear-gradient run.**

```
# Design heuristics

**Topology**: `opposing`
**Pillar config**: `none`
**Chamber height H**: `200.0 μm`
**Target profile**: `{'kind': 'linear_gradient', 'axis': 'x', 'c_high': 1.0, 'c_low': 0.0}`

## Dominant parameters (global sensitivity)
- `Q_total`, `W`

## Parameters that can be held loosely
- `theta`

## Tightest fabrication tolerances
| Parameter | −Δ (phys) | +Δ (phys) |
| `Q_total` |  4 μL/min |  5 μL/min |
| `W`       |  120 μm   |  135 μm   |
```

**How to use this.** Show it to whoever runs the SLA printer and syringe
pumps.  If the printer's reproducibility on `W` is ±50 μm, that comfortably
satisfies the "±120 μm" tolerance — you will get the target field.  If
the target were a sharp `step` profile, the `W` tolerance might drop to
`±20 μm` — now you have to measure the printed channel width before
trusting the chip.

### 7.3 Why GP-based and not CFD-based

Running 1024 extra CFD simulations per winner would take ~15 hours.
Sampling the trained GP takes seconds.  The GP is a measured surrogate
over the sampled region (its accuracy is implicitly calibrated by the BO
itself), so using it here is principled.  The safety net is that you
can always validate the top-k and bottom-k tolerances with CFD re-runs
(see `validate_with_cfd` in `ooc_optimizer/interpretability/tolerance.py`);
disagreements beyond the GP error bar mean the sampling should be
expanded.

---

## 8. 3D validation — did the 2D shortcut hold?

Every BO evaluation ran in 2D for speed.  Before you stake a paper or a
chip order on the winner, re-run it in 3D and check that the L² distance
to the target is still small.

```bash
tumor-chip validate-3d \
    --config examples/tumor_chip_linear_gradient/config.yaml \
    --bo-state examples/tumor_chip_linear_gradient/data/results/bo_opposing_none_H200 \
    --case-2d examples/tumor_chip_linear_gradient/data/cases/<winning_case_dir> \
    --output examples/tumor_chip_linear_gradient/data/validation_3d \
    --nz 20
```

`<winning_case_dir>` is the `case_dir` field of the best evaluation in
`evaluations.json`; the console output from the initial BO run printed
its full path.

### 8.1 What you should see

Inside `.../validation_3d/figures/` you will find three figures:

- `concentration_residual_3d_vs_2d.png` — side-by-side contour maps plus
  a residual column.  **The residual column should be predominantly
  pale.**  Absolute residuals above 0.3 mean the 2D-during-BO
  approximation was too loose and the BO may have settled on a geometry
  that is 2D-optimal but 3D-suboptimal.
- `centerline_3d_vs_2d.png` — `C(x, W/2)` overlay of target, 2D, and 3D.
  All three lines should lie within `0.1` of one another.
- `wss_scatter_bland_altman_v2.png` — retained v1 WSS check; the 2D
  proxy `τ = 6μU/H` should track the 3D-resolved `wallShearStress`
  within ±30 %.

And in `.../validation_3d/validation_3d_summary.json`:

```json
[
  {
    ...
    "L2_to_target_3d": 0.162,
    "L2_to_target_2d": 0.124,
    "L2_relative_delta": 0.30,   // ← the number that matters
    "tau_floor_mean_3d": 0.43,
    "converged_U": true,
    "converged_C": true
  }
]
```

**Interpretation of `L2_relative_delta`**:

| Value            | What to do                                                      |
|------------------|-----------------------------------------------------------------|
| `≤ 0.10` (10 %)  | 2D-during-BO was valid.  Proceed to fabrication.                |
| `0.10 – 0.30`    | Acceptable but report honestly in the paper.                    |
| `> 0.30`         | 2D is misleading this topology; re-run the BO in 3D for this target.  Budget cost is ≈ 4× the 2D cost. |

---

## 9. Running the full campaign (weekend-scale)

When you are ready to run the full paper-grade campaign, swap the
example config for the repository default and drop the `--single-target`
flag:

```bash
tumor-chip optimize \
    --config configs/default_config.yaml \
    --summary-out data/results/optimization_summary.json
```

**What this does.**
1. Sweeps all 24 discrete configurations on the primary target
   (linear gradient) = 24 BO runs × 60 evals ≈ 1,440 CFD evaluations.
2. Picks the winning topology and runs *only that topology* (all 4
   pillar configs × 2 heights) on the two secondary targets (bimodal,
   step).
3. Writes a single `optimization_summary.json` that ties the three target
   winners together.

**Wall-clock budget.** ~12 hours on 8 cores for the whole thing.  The
first 24 BO runs can be parallelised — add `--parallel` to fan them out
across cores:

```bash
tumor-chip optimize --config configs/default_config.yaml --parallel
```

**When to stop and rerun.** If during the sweep you see many
`mesh_ok=false` records piling up in one configuration, that topology is
producing invalid meshes at the current parameter bounds.  Narrow the
bounds (edit `continuous_bounds` in the config) and restart that
configuration alone.  You do not need to re-run the others.

### 9.1 Interpret all winners at once

After the full sweep:

```bash
tumor-chip interpret --results-dir data/results --sobol-n 1024
```

This produces one `interpretability/` folder per BO state dir, plus an
aggregated `data/results/interpretability_index.json`.  Compare the
`design_heuristics.md` files — the set of "dominant parameters" should
overlap across targets but the *tolerance intervals* will differ.  That
contrast is the core methodological claim of the paper.

### 9.2 Validate all winners in 3D

```bash
tumor-chip validate-3d \
    --config configs/default_config.yaml \
    --orchestrator-summary data/results/optimization_summary.json \
    --output data/validation_3d \
    --nz 25
```

One 3D run per target-winner is queued automatically; ~1.5 hour each on
8 cores.

---

## 10. Going to the bench — what changes in fabrication and imaging

This section is outside the scope of what the software itself runs, but
knowing it matters for interpreting pipeline output.  Full protocols are
in **Development Guide v2 §4.2 and §4.3**.

### 10.1 Printing the mould

- Read `design_heuristics.md` for the winner you intend to print.
- The tightest tolerance interval defines **which dimensions you must
  measure under the microscope** before trusting the chip.
- Example: if `W` has a ±50 μm tolerance and the SLA printer specification
  says ±30 μm reproducibility, fine — print and go.  If the printer is
  ±80 μm, you need to measure every printed mould and screen the outliers.

### 10.2 Chip lineup

The Development Guide prescribes 9 chips for the full paper (§4.2).  Chip
E is the critical one: it is printed with the **most-sensitive parameter
deliberately perturbed** by the predicted tolerance.  If the measured
concentration field degrades by the predicted amount (within ±20 %), the
interpretability prediction is validated.  If it degrades much more or
much less, the tolerance intervals were wrong — report that honestly.

### 10.3 Fluorescein imaging

Protocol summary:

1. Prime chip with DI water.
2. Switch drug inlet to `0.1 mg/mL fluorescein in PBS`; medium inlet to
   PBS.
3. Wait 5× the theoretical mean residence time for steady state.
4. Capture a long-exposure UV-illuminated top-down image.
5. Normalise pixel intensity against (a) a dark frame and (b) a
   saturated-chamber reference.

Compare the resulting `C_norm(x, y)` against the CFD-predicted `C(x, y)`
for the same chip (use `plot_concentration_contour` on the winner's
case).  Primary quantitative metric: pixel-wise L² distance.  Target is
`< 0.15` for the three target-profile winners and `> 0.25` for the
baseline (chip D).

---

## 11. Troubleshooting — where to look when something fails

### 11.1 OpenFOAM command not found

```
RuntimeError: OpenFOAM not found. Install via 'brew install gerlero/openfoam/openfoam@2406'
```

- **macOS**: `brew install gerlero/openfoam/openfoam@2406` and make sure
  `openfoam2406` is on your `PATH` (`which openfoam2406`).
- **Linux**: follow the OpenCFD install notes at
  <https://www.openfoam.com/download/> or use the Docker image
  (`docker compose run --rm tumor-chip-shell`).

### 11.2 A BO iteration returns `penalty_L2` (99.0) and the CFD log looks fine

Open the case's `scalarTransportFoam.log` first, then `simpleFoam.log`.
Common failure modes:

- `checkMesh` reports concave cells → snappyHexMesh parameters need
  tuning for this pillar configuration.  Edit the `snappy_*` keys in
  `solver_settings` of your config and re-run.  See `tip.md` entry
  "Pillar meshing quality policy".
- Residuals plateau above `1e-6` → solver did not converge.  Raise
  `max_iterations` in `solver_settings` or relax the residual control in
  `ooc_optimizer/cfd/template/system/fvSolution`.  See `tip.md` entry
  "Convergence threshold mismatch".
- `codedFixedValue` errors on macOS → do not use `codedFixedValue` in
  templates (the DMG-mounted OpenFOAM can't compile runtime code).  See
  `tip.md` entry "OpenFOAM macOS".

### 11.3 `cadquery` install fails

```
ImportError: cannot import name 'xxx' from 'OCP'
```

CadQuery pulls OCCT via `pip`.  On Linux install the system GL libraries
(`apt-get install -y libgl1`).  On macOS, reinstall in a clean conda env.

### 11.4 `pydantic.ValidationError: ... Missing required field`

Your YAML config is missing a field that the v2 schema expects.  The
normaliser (`ooc_optimizer.config.schema._normalize_legacy_fields`) fills
in defaults for the common v1-compatibility cases (`Q` → `Q_total`,
absent `r_flow`, absent `delta_W`, absent `inlet_topology`), but the
`paths` block always needs at least a `template_case`.  Copy
`configs/default_config.yaml` as a known-good starting point.

### 11.5 The BO finishes but `best_feasible` is `None`

All evaluations violated at least one constraint.  The two common causes:

1. **`τ_mean_min` too high for the flow-rate range.**  If the chamber is
   wide and `Q_total` is low, `τ_mean` cannot exceed `tau_mean_min`.
   Lower `tau_mean_min` in `optimization.constraints` or raise the
   `Q_total.min` bound.
2. **`f_dead_max` too tight for the pillar configuration.**  Dense pillar
   grids create recirculation zones.  Raise `f_dead_max` to ~0.10 for
   pillared configurations, or drop those from the sweep.

### 11.6 Everything ran but the figures look weird

Look at `case_dir/simpleFoam.log` and `case_dir/scalarTransportFoam.log`
for the winner.  Is `Solving for T` present?  Is the final residual of
`T` below `1e-6`?  If not, the scalar solver did not converge and the
winner's concentration field is untrustworthy — increase `end_time` in
`ooc_optimizer/cfd/scalar.py::set_scalar_controldict` (default 500) to
give scalarTransportFoam more iterations.

### 11.7 Where to look for known-good behaviour

Beside this guide, the canonical sources are:

- `tip.md` — per-bug debug notes, updated every time something breaks.
- `Development_Guide_v2.md` — science rationale and module specs.
- `CHANGELOG.md` — what changed in each release.
- `docs/concepts/` — short explanatory pages rendered at the docs site
  (`mkdocs serve --config-file docs/mkdocs.yml` to preview locally).

---

## 12. Glossary of files and folders

```
tumor-chip-design/
├── README.md                   short project summary + revision history
├── USER_GUIDE.md               this file
├── Development_Guide_v2.md     science rationale + module specs (READ ONCE)
├── CHANGELOG.md                what changed in each release
├── tip.md                      per-bug debug notes
├── LICENSE                     MIT
├── CITATION.cff                machine-readable citation metadata
├── pyproject.toml              packaging + ruff + mypy + pytest-cov config
├── configs/
│   ├── default_config.yaml     FULL v2 config — use this for the paper run
│   └── examples/               per-target-profile override snippets
├── ooc_optimizer/              implementation package (primary)
│   ├── geometry/               Module 1.2 — parametric chips + blockMesh
│   ├── cfd/                    Modules 1.1 / 2.2 — simpleFoam + scalarTransportFoam
│   ├── config/                 Module 2.3 — pydantic schema + JSONL logger
│   ├── optimization/           Modules 3.1 — BO loop + target profiles
│   ├── analysis/               Module 3.2 — concentration-field plots
│   ├── interpretability/       Module 3.3 — Sobol + GP grads + tolerances
│   ├── validation/             Module 4.1 — 3D CFD + comparison plots
│   └── cli.py                  the 'tumor-chip' console-script entry
├── tumor_chip_design/          public namespace that re-exports ooc_optimizer
├── scripts/                    thin driver scripts (wrapped by the CLI)
├── tests/                      pytest suite (101 green, 21 pre-existing stubs)
├── examples/
│   ├── tumor_chip_linear_gradient/   15-min worked example for new users
│   ├── tumor_chip_bimodal/           rim + rim target sibling
│   └── wss_uniformity/               v1 WSS example (engine-generality demo)
├── docker/                     Dockerfile + docker-compose.yml
├── docs/                       mkdocs-material site (docs.yml deploys to Pages)
├── paper/joss/                 JOSS companion paper draft
├── .github/                    CI / docs / release workflows, issue + PR templates
└── data/                       created at run time — DO NOT COMMIT
    ├── cases/                  OpenFOAM per-evaluation cases
    ├── results/                BO state + interpretability artefacts
    ├── validation_3d/          3D validation output
    └── figures/                publication-quality figures
```

---

## What a week of work looks like

| Day | What you do                                                                       | Where output lands                            |
|-----|-----------------------------------------------------------------------------------|-----------------------------------------------|
| 1   | §2 install, §4 smoke tests                                                        | `data/scalar_verification/`, `data/verification/` |
| 2   | §5 run the 15-minute example, §6 read the results                                 | `examples/tumor_chip_linear_gradient/data/`   |
| 2   | §7 read `design_heuristics.md`, present to the biology lead                       | `.../interpretability/`                       |
| 3   | §9 launch the full campaign with `--parallel` overnight                            | `data/results/`                               |
| 4   | §9.1 aggregate interpretability, §9.2 queue 3D validation                          | `data/validation_3d/`                         |
| 5   | §10 print 9 moulds, cast PDMS, bond chips                                         | wet-lab notebook                              |
| 6–7 | §10 fluorescein imaging of all 9 chips, compare to CFD predictions                 | wet-lab notebook + `data/experimental/`       |
| 8   | Draft the manuscript figures from the JSONL logs (`scripts/rebuild_figures.py`)    | `figures/`                                    |

At the end of day 7, you know which chamber geometries match each of the
three target profiles, which of their geometric features actually matter,
how precisely each must be held during fabrication, and whether the
fabrication tolerances you predicted are real.

That's the pipeline.  Good luck.
