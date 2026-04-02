# OoC-Optimizer: Development Guide

## Project Overview (Non-Technical)

This project builds an automated pipeline that designs, simulates, and physically validates microfluidic organ-on-chip (OoC) devices. The goal is to find the best chip geometry — the shape of tiny fluid channels — so that cells cultured inside experience uniform, physiologically realistic fluid shear stress.

The pipeline works like this: a computer generates thousands of candidate chip shapes, simulates fluid flow through each one, and uses a smart search algorithm (Bayesian optimization) to zero in on the best design. The winning design is then 3D-printed as a mold, cast in silicone (PDMS), and tested with real fluids to confirm the simulation predictions match reality.

The target application is vascular endothelial cell culture. These cells line blood vessels and are extremely sensitive to wall shear stress (WSS). Non-uniform shear produces inconsistent cell behavior, which ruins experimental results. Our optimizer eliminates this problem by engineering uniform shear across the entire culture surface.

The deliverable is an open-source tool and an academic publication, or research writeup, demonstrating that the optimized chip outperforms a naive baseline design in both simulation and physical experiment.

---

## Target Scope

**Single biological target:** Vascular endothelial culture (WSS = 0.5–2.0 Pa).

**Optimization approach:** 2D CFD only during optimization; 3D CFD and physical experiments for post-hoc validation of the winner.

**Validation experiments:** Food dye flow visualization and fluorescein washout residence-time distribution (RTD). No micro-PIV or inline pressure sensors.

**Discrete variable handling:** Enumerate pillar configurations and chamber heights; run independent BO for each combination. No mixed-integer optimization.

**Total computational budget:** ~400 CFD runs across 8 discrete configurations (4 pillar layouts × 2 heights), each with 50 evaluations (15 Sobol + 35 BO iterations). Estimated ~3.3 hours on 1 core, ~25 min on 8 cores.

---

## Team Structure and Ownership

Two developers: **Dev A** (expert programmer + 3D printing access) and **Dev B** (intermediate programmer). Dev A owns all modules requiring deep OpenFOAM/BoTorch expertise or physical equipment. Dev B owns geometry generation, configuration, results analysis, and code packaging — work that is genuinely unblocked and feeds directly into the paper or final deliverable. Dev A's workload is heavily front-loaded with code (weeks 1–5) and shifts to physical work (weeks 6–8); Dev B's work is more evenly distributed. Both contribute to integration and testing, and both developers should understand the full pipeline.

---

## Architecture Overview

The pipeline is a linear chain of five stages, each consuming the output of the previous:

1. **Parametric Geometry Generator** → produces fluid domain STL + mold STL
2. **Automated CFD Engine** → meshes the domain, runs simulation, extracts metrics
3. **Bayesian Optimization Loop** → calls stages 1–2 as a black-box function, searches for optimal parameters
4. **3D CFD Validation** → high-fidelity confirmation of the 2D winner
5. **Fabrication and Experimental Validation** → physical chips, flow experiments, data analysis

A shared configuration schema (YAML or JSON) passes parameters between stages. Every stage must be fully automated with no manual intervention per evaluation.

---

## Module Specifications

Modules are listed in chronological development order. Modules within the same phase can be developed in parallel by the two developers.

---

### Phase 1: Foundation (Weeks 1–2)

These modules establish the two independent pillars of the pipeline. They have zero dependencies on each other and should be developed simultaneously.

---

#### Module 1.1 — CFD Template and Solver Verification

**Owner:** Dev A

**Purpose:** Establish a verified OpenFOAM simulation environment and confirm it produces correct results on a known analytical case before connecting it to anything else.

**Technical Details:**

The solver is `simpleFoam` (steady-state SIMPLE algorithm for incompressible laminar flow). All optimization runs use 2D simulations implemented as a single-cell-thick slab in the z-direction with `empty` boundary conditions on front/back faces.

The critical subtlety: a 2D plan-view simulation computes depth-averaged velocity, not resolved floor WSS. Floor WSS must be estimated from the analytical parabolic profile assumption: `τ_floor = 6 μ U_avg / H`, where `U_avg` is the local depth-averaged velocity and `H` is the channel height. This approximation is valid for low Reynolds number flow with channel aspect ratio W/H > 3, which holds throughout our parameter space.

Verification task: simulate Poiseuille flow in a straight rectangular channel with known dimensions. Compare the simulated centerline velocity and computed WSS against the analytical solution. Agreement within 2% confirms the solver, mesh resolution, and post-processing are correct.

**Boundary conditions for all simulations:**

| Patch | Velocity | Pressure |
|-------|----------|----------|
| inlet | fixedValue (parabolic profile, U_mean = Q / A_inlet) | zeroGradient |
| outlet | zeroGradient | fixedValue (0) |
| walls | noSlip | zeroGradient |
| floor (culture surface) | noSlip | zeroGradient |
| frontAndBack | empty | empty |

Convergence criterion: residuals < 1e-6 for both U and p. Typical convergence: 200–500 iterations.

**Deliverables:**
- A template OpenFOAM case directory with correct boundary conditions, solver settings, and convergence criteria
- A mesh convergence study on the rectangular channel (1×, 2×, 4× refinement) showing WSS converges to <2% change between 2× and 4×
- Documented verification results (analytical vs. simulated)

---

#### Module 1.2 — Parametric Geometry Generator

**Owner:** Dev B

**Purpose:** Build a CadQuery-based geometry engine that takes a parameter vector and produces two watertight STL files: the fluid domain (for CFD) and the mold negative (for 3D printing).

**Technical Details:**

Input: parameter vector `x = (W, d_p, s_p, θ, Q)` plus discrete configuration (pillar layout, H).

Output: `fluid_domain.stl` (internal fluid volume) and `chip_mold.stl` (mold for PDMS casting).

The chip architecture is a single rectangular culture chamber (length L = 10 mm, fixed) with one inlet and one outlet, each connected by tapered expansions/contractions. An optional array of cylindrical micropillars inside the chamber redistributes flow.

**Continuous parameters (optimized by BO):**

| Variable | Symbol | Range | Units | Notes |
|----------|--------|-------|-------|-------|
| Chamber width | W | 500–3000 | μm | Culture area width |
| Pillar diameter | d_p | 100–400 | μm | Constrained by SLA printer minimum feature ≥100 μm |
| Pillar gap (center-to-center) | s_p | 200–1000 | μm | Must satisfy s_p > d_p + 100 μm |
| Inlet taper angle | θ | 15–75 | degrees | Gradual taper reduces inlet jet effect |
| Flow rate | Q | 5–200 | μL/min | Operating condition, not a geometric parameter |

**Discrete parameters (enumerated):**

| Variable | Levels |
|----------|--------|
| Pillar configuration | {none, 1×4, 2×4, 3×6} |
| Chamber height | {200, 300} μm |

**Fixed parameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Chamber length L | 10 mm | Standard OoC culture area |
| Inlet/outlet channel width W_in | 500 μm | Standard tubing ID |
| Fluid viscosity μ | 0.001 Pa·s | Culture media ≈ water |
| Fluid density ρ | 1000 kg/m³ | Culture media ≈ water |

Pillar placement: evenly spaced in a grid pattern within the chamber. For an `R×C` configuration, pillar centers are at `cx = L * (j+1) / (C+1)` along length and `cy = W * (i+1) / (R+1)` along width.

The mold is computed as: bounding box (with 2 mm wall thickness on all sides) minus the fluid domain. Include alignment pin features for glass slide registration.

**Geometry validation checks (run automatically on every generated geometry):**
- Volume is within expected range for the given parameters
- No self-intersecting faces (CadQuery `isValid()`)
- All features (gaps, diameters, wall thicknesses) exceed printer minimum of 100 μm
- Constraint `s_p > d_p + 100 μm` is satisfied (pillars don't overlap or nearly touch)

**Units convention:** All internal geometry math in μm. Convert to mm for STL export (divide by 1000). OpenFOAM and 3D printers both expect mm.

**Deliverables:**
- A Python module (`geometry.py` or similar) exposing a `generate_chip(params) -> (fluid_stl_path, mold_stl_path)` function
- Verified STL output for all 4 pillar configurations at representative parameter values
- Unit tests for constraint validation and edge cases (minimum/maximum parameter values)

---

### Phase 2: Pipeline Integration (Weeks 3–4)

These modules connect the geometry generator to the CFD solver and create the automated evaluation function that the optimizer will call.

---

#### Module 2.1 — Automated Meshing Pipeline

**Owner:** Dev A (with Dev B providing STL inputs for testing)

**Purpose:** Take a fluid domain STL and automatically produce a valid OpenFOAM mesh, with no human intervention.

**Technical Details:**

Two meshing strategies depending on geometry:

1. **No pillars:** Use `blockMesh` to generate a structured quad mesh. Target ~12,000 cells (200 along length × 60 across width). Runs in ~5–15 seconds.

2. **With pillars:** Use `cfMesh` (preferred) or `snappyHexMesh` to handle cylindrical cutouts. Target ~30,000 cells. Runs in ~20–40 seconds. `cfMesh` is preferred because `snappyHexMesh` can fail on tight geometries near pillars.

For the 2D approach: the mesh is one cell thick in z, with `empty` type on front and back patches. The mesher must handle the STL import, patch naming (inlet, outlet, walls, floor, frontAndBack), and produce a mesh that passes `checkMesh` with no fatal errors.

A fallback mechanism is needed: if the primary mesher fails (returns non-zero exit code or `checkMesh` reports severe errors), log the failure and return a penalty value to the optimizer rather than crashing the loop.

**Deliverables:**
- A mesh generation script that accepts an STL path and outputs a valid OpenFOAM `polyMesh` directory
- Automatic patch identification and naming
- Mesh quality validation via `checkMesh`
- Fallback/error handling for meshing failures

---

#### Module 2.2 — CFD Run Automation and Metric Extraction

**Owner:** Dev A

**Purpose:** Wrap the full sequence of mesh → solve → post-process into a single callable function that returns optimization metrics from a completed simulation.

**Technical Details:**

The function takes a parameter vector, calls Module 1.2 to generate geometry, calls Module 2.1 to mesh it, sets boundary conditions (inlet velocity from Q and A_inlet), runs `simpleFoam`, then extracts metrics.

Metrics to extract from each completed simulation:

| Metric | Definition | Role |
|--------|-----------|------|
| CV(τ) | σ(τ_floor) / μ(τ_floor), coefficient of variation of WSS on the culture floor | **Primary objective** (minimize) |
| τ_mean | Area-weighted average WSS on the floor | Constraint: must be in [0.5, 2.0] Pa |
| f_dead | Fraction of floor area where local velocity < 0.1 × mean velocity | Constraint: must be < 0.05 |
| τ_min, τ_max | Min/max WSS on the floor | Diagnostic |
| ΔP | Pressure drop inlet to outlet | Diagnostic (not optimized) |
| converged | Boolean: did residuals reach the convergence criterion? | Gate: if false, return penalty |

WSS on the floor is computed as `τ = 6 μ U_local / H` from the depth-averaged velocity field, not from the OpenFOAM `wallShearStress` function object (which reports the 2D in-plane wall shear, not the floor shear in a plan-view 2D sim).

If the simulation fails to converge or the mesher fails, return a large penalty value for CV(τ) (e.g., 999.0) and mark constraints as violated. This tells the optimizer to avoid that region of parameter space.

**Deliverables:**
- A Python function `evaluate_cfd(params, pillar_config, H) -> metrics_dict` that orchestrates the full geometry → mesh → solve → extract pipeline
- JSON output of metrics for each evaluation (for logging and later analysis)
- Robust error handling: meshing failures, solver divergence, and timeout (kill runs exceeding 5 minutes)

---

#### Module 2.3 — Configuration Schema and Logging

**Owner:** Dev B

**Purpose:** Define a shared configuration format and a structured logging system so that every evaluation is fully reproducible and traceable.

**Technical Details:**

A YAML configuration file specifies:
- All fixed parameters (L, W_in, μ, ρ)
- Parameter bounds for continuous variables
- Discrete variable levels
- Solver settings (convergence criterion, max iterations, mesh resolution)
- Paths (template case, output directory, STL export directory)

Every CFD evaluation should produce a log entry containing: the input parameter vector, the discrete configuration, the output metrics, wall-clock time, convergence status, and the case directory path. Store evaluations in a CSV or JSONL file for later analysis and BO convergence plotting.

**Deliverables:**
- A config schema (YAML with comments) and a loader module
- An evaluation logger that appends to a structured log file
- A utility to replay/inspect any past evaluation from the log

---

### Phase 3: Optimization (Weeks 4–5)

---

#### Module 3.1 — Bayesian Optimization Loop

**Owner:** Dev A

**Purpose:** Implement the BO loop using BoTorch that calls the automated evaluation function and searches for the parameter vector minimizing CV(τ) subject to constraints.

**Technical Details:**

Framework: BoTorch (PyTorch-based). The GP surrogate uses the Matérn 5/2 kernel (BoTorch default), with length scale and output scale learned via marginal likelihood maximization.

Initialization: 15 Sobol quasi-random points (3× the 5 continuous dimensions). Sobol sequences have better space-filling properties than pure random sampling.

Acquisition function: `ConstrainedExpectedImprovement`, which weights expected improvement by the probability of constraint satisfaction. Three constraint GPs are trained independently:
- `c1 = τ_mean - 0.5` (feasible if ≥ 0, i.e., τ_mean ≥ 0.5 Pa)
- `c2 = 2.0 - τ_mean` (feasible if ≥ 0, i.e., τ_mean ≤ 2.0 Pa)
- `c3 = 0.05 - f_dead` (feasible if ≥ 0, i.e., f_dead ≤ 5%)

Budget per discrete configuration: 50 evaluations (15 initial + 35 BO iterations). Total across 8 configurations: 400 evaluations.

The 8 configurations are independent and embarrassingly parallel — run them sequentially or in parallel depending on available cores. Each is a self-contained BO run.

After all 8 runs complete, select the overall winner: the configuration and parameter vector with the lowest CV(τ) among all feasible solutions across all runs.

**Design rationale for single-objective:** The original plan used a weighted sum of CV, dead zone fraction, and pressure drop. This is problematic because it couples three different phenomena into one scalar, making it hard to interpret what the optimizer is doing. Instead: CV(τ) is the sole objective, dead zone and physiological range are hard constraints, and pressure drop is dropped entirely (at these flow rates, ΔP is always < 1 kPa, well within syringe pump capacity).

**Deliverables:**
- A BO runner module that executes the full loop for one discrete configuration
- A top-level orchestrator that runs all 8 configurations and selects the winner
- Convergence logging: objective value vs. iteration, best feasible value vs. iteration
- Serialization of the GP model and all evaluation data for later inspection

---

#### Module 3.2 — Results Analysis and Visualization

**Owner:** Dev B

**Purpose:** Generate analysis plots from BO results to characterize optimizer behavior and compare configurations.

**Technical Details:**

Plots to generate:
- BO convergence curves: CV(τ) vs. iteration for each of the 8 configurations, showing improvement from Sobol initialization to final optimum
- Parameter heatmaps: how the optimized parameters differ across pillar configurations
- WSS contour maps (2D): color maps of τ on the culture floor for the baseline geometry (high CV, central jet) vs. the optimized geometry (low CV, uniform). These are the key paper figures.
- Constraint satisfaction: scatter plot of τ_mean and f_dead for all evaluated points, colored by feasibility

The baseline geometry for comparison: flat rectangular chamber, no pillars, no inlet taper (θ = 90°), W = 1500 μm, H = 200 μm. This represents "what an engineer would design without optimization." Literature consistently shows this produces a central jet with high-shear center and low-shear edges.

**Deliverables:**
- Plotting scripts that consume the evaluation log and produce publication-quality figures
- Automated baseline evaluation for side-by-side comparison
- Summary table of best results per configuration

---

### Phase 4: Validation (Weeks 6–7)

---

#### Module 4.1 — 3D CFD Validation

**Owner:** Dev B

**Purpose:** Run a high-fidelity 3D simulation of the optimized geometry to verify that the 2D approximation was valid and to generate ground-truth flow fields for comparison with experiments.

**Technical Details:**

Mesh: `snappyHexMesh` from the optimized geometry's STL. Target 500k–1M cells with 5-layer boundary layer refinement at the floor (first layer height ~2 μm for resolved WSS). Expected run time: 5–30 minutes depending on mesh size.

Solver: `simpleFoam` (same as 2D, but now fully 3D).

Comparison metrics:
- 2D-predicted CV(τ) vs. 3D-predicted CV(τ): quantify the approximation error
- Scatter plot of 2D vs. 3D WSS at matched floor locations + Bland-Altman analysis
- 3D WSS contour map on the culture floor (this becomes a key paper figure)
- Velocity streamlines at the mid-plane height (for comparison with experimental dye patterns)
- Pressure drop: 2D estimate vs. 3D resolved

Run the same 3D validation on the baseline geometry for a complete comparison.

**Deliverables:**
- 3D simulation results for both optimized and baseline geometries
- 2D vs. 3D comparison figures (scatter + Bland-Altman + contour maps)
- Quantified approximation error to report in the paper

---

#### Module 4.2 — Fabrication Protocol

**Owner:** Dev A

**Purpose:** Produce physical chips from the optimized and baseline geometries using SLA 3D printing and PDMS casting.

**Technical Details:**

Fabrication route: SLA mold → PDMS cast → glass slide bonding.

Chip lineup:

| Chip | Geometry | Replicates | Purpose |
|------|----------|-----------|---------|
| A | Optimized (BO winner) | 2 | Demonstrate pipeline works |
| B | Baseline (no pillars, 90° inlet) | 2 | Comparator |
| C | Second-best from BO | 1 | Show optimizer explores meaningfully different designs |

Total: 5 chips, approximately 2 days of fabrication.

**Fabrication steps:**

1. **STL export:** Export `chip_mold.stl` from Module 1.2, verify dimensions in mm, confirm 2 mm wall thickness on all sides, include alignment pin features for glass registration.

2. **SLA printing (1–2 hours per batch):** Layer height 50 μm, XY pixel ≤ 50 μm. Use standard clear resin. Print orientation: face-down (channel features on build plate) for best surface finish on the critical mold surface. Minimal supports, positioned away from channel features.

3. **Post-processing mold (30 min):** IPA wash (2×5 min), UV post-cure (10 min). Optional: thin epoxy coating (XTC-3D thinned 3:1 with acetone) for smoother surface. Optional: silane release agent for easier PDMS demolding.

4. **PDMS casting (3.5 hours):** Sylgard 184, 10:1 base-to-curing agent ratio, ~5 mL per chip. Pour over mold, degas in vacuum desiccator for 30 min, cure at 65°C for 2 hours. Peel PDMS slab from mold.

5. **Chip assembly (15 min per chip):** Punch inlet/outlet holes with 1 mm biopsy punch. Clean PDMS and glass slide. Activate both surfaces with handheld corona treater. Press PDMS onto glass within 30 seconds of activation. Bake at 65°C for 10 min. Insert Tygon tubing (OD 1.5 mm), seal with UV adhesive if needed.

6. **Connect to syringe pump:** Luer-lock adapters, prime tubing with DI water to remove air.

**Deliverables:**
- 5 assembled, leak-tested chips ready for experiments
- Documented fabrication log with any deviations or issues

---

#### Module 4.3 — Flow Experiments and Data Analysis

**Owner:** Dev A

**Purpose:** Run physical flow experiments on the fabricated chips and compare results to CFD predictions.

**Technical Details:**

**Experiment 1 — Food dye flow visualization (15 min per chip):**

Protocol: Set syringe pump to design flow rate Q*. Fill chip with DI water until steady state. Switch to diluted food dye (1:20 in water). Record top-down video with smartphone on tripod over white backlight. Capture steady-state still image. Reverse flush with DI water.

Analysis: Trace dye front progression frame-by-frame, compare to CFD streamlines. Compare steady-state dye intensity map to CFD velocity magnitude contour. Identify dead zones (last-to-fill regions), compare to CFD-predicted f_dead regions. Side-by-side figure: optimized vs. baseline at the same flow rate.

**Experiment 2 — Fluorescein washout RTD (30 min per chip):**

Protocol: Fill chip with 0.1 mg/mL fluorescein sodium in PBS at design flow rate. Under UV lamp (365 nm) with blue-light filter on camera, start video recording. Switch inlet to pure PBS. Record washout for 5× the theoretical mean residence time (τ_res = V_chip / Q).

Analysis (Python + OpenCV): Define 4–6 ROIs (inlet region, mid-chamber, near pillars, outlet, dead zone). Track mean fluorescence intensity per ROI vs. time. Normalize: `I(t) = (I_raw(t) - I_background) / (I_initial - I_background)`. Plot `I(t)/I(0)` vs. `t/τ_res` for each ROI.

Expected results: well-mixed regions show exponential decay; dead zones show slow tails. Optimized chip should show faster, more uniform washout across all ROIs.

**Quantitative comparison metrics:**

| Metric | Measurement method | Expected: Optimized vs. Baseline |
|--------|--------------------|----------------------------------|
| Dye front uniformity | Width of dye front at 50% chamber length | Lower (more uniform) |
| Dead zone fill time | Time for last ROI to reach 90% steady-state | Shorter |
| Washout half-life CV | CV of half-life across ROIs | Lower |
| Washout tail ratio | I(3×τ_res)/I(0) averaged across ROIs | Lower |

**Deliverables:**
- Raw experimental data (video files, extracted intensity curves)
- Processed comparison figures (optimized vs. baseline for both experiments)
- Quantitative metrics table for all 5 chips

---

### Phase 5: Documentation and Publication (Weeks 8–10)

---

#### Module 5.1 — Paper Figures

**Owner:** Both (Dev A: CFD figures; Dev B: experimental figures)

**Purpose:** Generate all publication-quality figures.

**Figure list:**

1. Pipeline schematic — block diagram: specification → CadQuery → OpenFOAM → BoTorch → STL → SLA → PDMS → experiment
2. Mesh convergence study — WSS vs. refinement level for baseline geometry
3. BO convergence — CV(τ) vs. iteration, one panel per pillar configuration
4. Optimized vs. baseline geometry — side-by-side CAD renders
5. 2D CFD WSS contour maps — baseline (central jet, high CV) vs. optimized (uniform, low CV). Key figure.
6. 2D vs. 3D validation — scatter plot + Bland-Altman for the optimized geometry
7. Experimental photos — food dye in optimized vs. baseline at steady state, visually matched to CFD streamlines
8. Washout curves — fluorescence decay at multiple ROIs, optimized vs. baseline
9. Summary bar chart — CV(τ), f_dead, washout uniformity for baseline vs. optimized with error bars from replicates

---

#### Module 5.2 — Code Packaging and Repository

**Owner:** Dev B

**Purpose:** Package the full pipeline as a reproducible, installable open-source tool.

**Technical Details:**

Repository name: `ooc-optimizer`. Structure should include: the geometry generator, CFD automation scripts, BO loop, post-processing and plotting utilities, example configurations, and a README with installation and usage instructions.

Dependencies to document: CadQuery, OpenFOAM (specify version), BoTorch/PyTorch, NumPy, Matplotlib, OpenCV (for experimental analysis).

Include a minimal example: a single BO run on the simplest configuration (no pillars, H=200) that completes in ~30 minutes on a laptop.

---

#### Module 5.3 — Manuscript

**Owner:** Both

**Purpose:** Write the paper targeting Biomicrofluidics or Lab on a Chip.

Recommended title: *"Bayesian Optimization of Microfluidic Organ-on-Chip Geometry for Wall Shear Stress Uniformity with Rapid Prototyping Validation"*

Companion software paper for JOSS: *"OoC-Optimizer: An Open-Source Pipeline for Automated Design, Simulation, and Fabrication of Organ-on-Chip Microfluidic Devices"*

---

## Development Schedule

| Week | Dev A | Dev B | Milestone |
|------|-------|-------|-----------|
| 1 | M1.1: OpenFOAM setup, Poiseuille verification, mesh convergence | M1.2: CadQuery parametric model, STL export for all pillar configs | Foundation verified independently |
| 2 | M1.1: Finalize template case, document verification results | M1.2: Mold geometry, constraint validation, unit tests | Both pillars complete |
| 3 | M2.1: Automated meshing for no-pillar and pillar geometries | M2.3: Config schema, evaluation logger | Meshing and config ready |
| 4 | M2.2: Full evaluate_cfd function, error handling, timeout | M2.3: Finish logger; begin M3.2 plotting framework | End-to-end evaluation works |
| 5 | M3.1: BO loop, run all 8 configurations | M3.2: Convergence plots, WSS contour maps, analysis | Optimization complete |
| 6 | M4.2: Print molds, cast PDMS, assemble 5 chips | M4.1: 3D CFD validation (optimized + baseline) | Validation in parallel |
| 7 | M4.3: Run experiments 1–2 on all chips | M4.1: 2D vs. 3D comparison figures | Physical data collected |
| 8 | M4.3: Analyze experimental data; M5.1: CFD figures (figures 1–6) | M3.2: Final figures; M5.1: experimental figures (7–9) | All figures drafted |
| 9 | M5.3: Begin manuscript draft | M5.2: Code packaging, repo, README | Repo publishable |
| 10 | M5.3: Manuscript review and revision | M5.3: Manuscript review and revision | Submission-ready draft |

Buffer: If fabrication or experiments slip in weeks 6–7, week 8 absorbs the delay and writing shifts by 1 week.

