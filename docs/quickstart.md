# 15-minute quickstart

This page reproduces a minimal inverse-design run end-to-end: solver
verification → tiny BO run → interpretability report.  It targets the
Docker image so you do not need OpenFOAM on the host.

## 0. Clone and start the container

```bash
git clone https://github.com/ooc-loop/tumor-chip-design.git
cd tumor-chip-design
docker compose -f docker/docker-compose.yml run --rm tumor-chip-shell
```

Everything below runs inside the container's `/workspace` mount, which is
just your clone.  Exiting the shell leaves all results on the host.

## 1. Verify the scalar solver (~2 min)

```bash
tumor-chip verify-scalar \
    --output data/scalar_verification \
    --n-cells 100
```

Expected: all four Peclet numbers (1, 10, 100, 1000) report `L2_rel < 0.02`
against the analytic advection-diffusion solution.  The results and case
directories live under `data/scalar_verification/`.

## 2. Run a tiny BO sweep (~8 min)

Copy the default config and shrink it for the quickstart:

```bash
cp configs/default_config.yaml configs/quickstart.yaml
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/quickstart.yaml"))
cfg["optimization"]["n_sobol_init"] = 4
cfg["optimization"]["n_bo_iterations"] = 4
cfg["discrete_levels"]["pillar_config"] = ["none"]
cfg["discrete_levels"]["chamber_height"] = [200]
cfg["discrete_levels"]["inlet_topology"] = ["opposing"]
yaml.safe_dump(cfg, open("configs/quickstart.yaml", "w"))
PY

tumor-chip optimize \
    --config configs/quickstart.yaml \
    --single-target \
    --summary-out data/results/quickstart_summary.json
```

This runs 8 CFD evaluations on a single discrete configuration and prints
the winning chip geometry.

## 3. Interpret the winner (~30 s)

```bash
tumor-chip interpret \
    --results-dir data/results \
    --sobol-n 256
```

Each `bo_*` directory under `data/results` will receive an
`interpretability/` subfolder containing:

- `sobol.png` — Sₜ / S₁ bar chart per parameter
- `local_sensitivity.png` — ∂μ/∂x ranking at the BO optimum
- `tolerance.png` — per-parameter ± tolerance intervals
- `design_heuristics.md` — auto-written plain-English summary

Open `design_heuristics.md` in your editor — that's the core scientific
deliverable of the pipeline.

## 4. (Optional) 3D validation (~5 min)

If time allows, re-run the winner in 3D to check that the 2D-during-BO
approximation held:

```bash
tumor-chip validate-3d \
    --config configs/quickstart.yaml \
    --bo-state data/results/bo_opposing_none_H200 \
    --output data/validation_3d/quickstart \
    --nz 15
```

Look at `data/validation_3d/quickstart/figures/concentration_residual_3d_vs_2d.png`
— the residual column should be mostly < 0.1 in absolute concentration units.

---

**What's next?**

- Full BO campaign: drop `--single-target` to sweep all 24 discrete
  configurations and run the winning topology on all three target profiles.
- Custom target profiles: see
  [Inverse design](concepts/inverse-design.md) for the callable interface.
- Reproducing paper figures: `scripts/rebuild_figures.py` regenerates every
  figure from the evaluation logs in under 5 minutes.
