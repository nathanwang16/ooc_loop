---
title: "tumor-chip-design: Interpretable inverse design of tumor-on-chip chambers"
tags:
  - Python
  - microfluidics
  - organ-on-chip
  - tumor-on-chip
  - computational fluid dynamics
  - OpenFOAM
  - Bayesian optimization
  - interpretability
  - sensitivity analysis
  - inverse design
authors:
  - name: Dev A
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Dev B
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Affiliation TBD
    index: 1
date: 22 April 2026
bibliography: paper.bib
---

# Summary

`tumor-chip-design` is an open-source Python pipeline that automates the
inverse design of tumor-on-chip microfluidic chambers whose steady-state
concentration field matches a user-specified target profile, and that
reports a principled estimate of the fabrication tolerance each geometric
parameter needs.  It couples a CadQuery parametric geometry generator, an
OpenFOAM two-step CFD solve (momentum via `simpleFoam`, then a frozen-flow
passive-scalar solve via `scalarTransportFoam`), a BoTorch
constrained-Bayesian-optimisation loop, and a post-hoc interpretability
layer that combines Sobol indices, closed-form Gaussian-process gradients,
and bisection-on-GP tolerance intervals.

The package is deliberately general-purpose: the inner loop solves a
scalar transport problem on a geometry that is parametrised by a fixed
set of continuous + discrete variables, so the same engine can target any
quantity that is well approximated as a steady passive scalar on a
frozen velocity field.  A retained worked example
(`examples/wss_uniformity/`) demonstrates this by optimising wall-shear-
stress uniformity on the same geometric family.

# Statement of need

Tumor-on-chip devices rely on shaped spatial gradients of drugs, nutrients
and tracers to recapitulate the microenvironment of solid tumours
[@ayuso2020; @doherty2022].  Chamber-geometry-driven gradients are an
active area of device development, but the design process is typically
ad-hoc: a geometry is chosen by analogy to existing devices, simulated
or measured, and iterated.  There is no open-source pipeline that closes
the loop, and more importantly there is no open-source tool that answers
the question that is actually useful to other labs: *"given that this
geometry works, which of its features matter, and how precisely do I need
to hold each one?"*

`tumor-chip-design` fills that gap.  The BO machinery is standard
[@balandat2020botorch] and serves as a means to train an informed GP
surrogate; the novel scientific deliverable is the interpretability
output.  First- and total-order Sobol indices computed on the GP surrogate
via `SALib` [@herman2017salib] identify the parameters that control the
field globally; exact autograd-based gradients identify which parameters
dominate *at the optimum*; and bisection-on-GP along each axis produces a
per-parameter tolerance interval for the BO winner — the direct
fabrication-tolerance spec a wet lab needs.

The package enforces reproducibility by construction: every CFD run is
logged to JSONL, every BO run serialises its GP state and training data,
and a command (`tumor-chip`) regenerates every paper figure from the logs
without re-running the solvers.  A Docker image bundles OpenFOAM 2406,
Python 3.11 and the package so that a fresh clone reproduces the
manuscript results in a single command.

# Pipeline overview

![Pipeline schematic](figs/pipeline.png){ width=90% }

1. **Parametric geometry** — three inlet topologies (`opposing`,
   `same_side_Y`, `asymmetric_lumen`) and a 7-dimensional continuous
   parameter space.  Two-inlet topologies follow a Y-junction or
   Ayuso-2020-style lumen convention [@ayuso2020].
2. **Meshing** — topology-aware multi-block `blockMesh` in 2D for the BO
   inner loop; `snappyHexMesh` with a boundary-layer addition for the 3D
   validation step.
3. **CFD** — `simpleFoam` momentum solve followed by `scalarTransportFoam`
   on the frozen velocity field.
4. **Bayesian optimisation** — BoTorch SingleTaskGP (Matérn 5/2 kernel)
   with `ConstrainedExpectedImprovement`, minimising the relative L² between
   the achieved concentration field and a target profile
   (`linear_gradient`, `bimodal`, `step`, or any user-supplied callable),
   subject to physiological flow and dead-zone constraints.
5. **Interpretability** — Sobol indices + local GP gradients + tolerance
   intervals on the trained surrogate, with the option to validate the
   top-k and bottom-k tolerances with CFD re-runs.
6. **3D validation** — the BO winner is re-meshed and re-solved in 3D and
   the 2D-vs-3D L² delta is reported as a go / no-go signal for the BO
   fidelity choice.

# Typical use

```bash
tumor-chip optimize    --config configs/default_config.yaml
tumor-chip interpret   --results-dir data/results
tumor-chip validate-3d --config configs/default_config.yaml \
                       --orchestrator-summary data/results/optimization_summary.json \
                       --output data/validation_3d
```

After the pipeline finishes, every `bo_*` directory under `data/results`
contains an auto-generated `design_heuristics.md` that translates the
numeric interpretability output into a short plain-English note suitable
for a lab notebook.

# Architecture and extensibility

The package is laid out as a conventional Python project with a hatchling
backend and three optional-dependency extras (`dev`, `test`, `docs`).
The CI workflow (`.github/workflows/ci.yml`) exercises Python 3.10, 3.11
and 3.12.  Extending the package to a new target scalar requires only a
new `TargetProfile` callable; extending it to a new geometric topology
requires adding one function to `ooc_optimizer.geometry.topology_blockmesh`.

# Acknowledgements

TBD.

# References
