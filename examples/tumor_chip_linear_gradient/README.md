# Example: linear gradient

Smallest complete worked example of the v2 pipeline.  Single discrete
configuration (`opposing` topology, no pillars, `H = 200 μm`), 20
evaluations (8 Sobol + 12 BO), one target profile.

## Runtime

~15 minutes on 8 cores on a recent laptop, assuming OpenFOAM 2406 is on
`PATH` (or accessible via the `openfoam2406` wrapper).

## Run it

```bash
python examples/tumor_chip_linear_gradient/run.py
```

Artefacts end up under
`examples/tumor_chip_linear_gradient/data/{stl,cases,results,figures}/`.

## What to look at when it finishes

1. `data/results/bo_opposing_none_H200/evaluations.json`
   — full BO trace, training data, GP state.
2. `data/results/bo_opposing_none_H200/interpretability/design_heuristics.md`
   — auto-generated English summary of the dominant parameters + tolerance
   intervals.  **This is the headline deliverable.**
3. `data/results/bo_opposing_none_H200/interpretability/sobol.png`,
   `local_sensitivity.png`, `tolerance.png` — figures backing the note.
