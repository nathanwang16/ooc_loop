# Example: WSS uniformity (v1 retained)

This example is the v1 worked problem, retained verbatim as evidence that
the `tumor-chip-design` engine is general-purpose: swapping the target
scalar from concentration to floor wall-shear-stress requires only a
configuration change (different constraint bounds + the `cv_tau` objective
instead of `L2_to_target`), not a code change.

The v1 development guide lives at
[`Development_Guide.md`](../../Development_Guide.md); the v2 guide that
supersedes it lives at
[`Development_Guide_v2.md`](../../Development_Guide_v2.md).  Nothing under
this directory is required for the v2 manuscript; it exists to:

1. Provide a second worked example for the JOSS companion paper
   (demonstrates extensibility to other inverse-design problems).
2. Let reviewers run the original v1 campaign end-to-end without
   resurrecting the pre-v2 tree.

## Run

```bash
python examples/wss_uniformity/run.py
```

By default the example runs one discrete configuration (`topology =
opposing`, `pillar = none`, `H = 200 μm`) through the v2 BO orchestrator.
The objective value reported is the v1 `cv_tau` (CV of the floor wall-
shear-stress); `L2_to_target` is left NaN because no target profile is
meaningful here.

## Known quirks

- Under the v2 schema the "single-inlet" v1 chip is represented as an
  `opposing` topology with `r_flow = 0.5` and `delta_W → 0` — the two
  inlet patches degenerate to a single face in the limit.  The full
  v1 single-inlet geometry is preserved in the git history at tag
  `v0.5.0`.
- The WSS constraint window is tighter here (`τ ∈ [0.5, 2.0]` Pa) than in
  the v2 default (`τ ∈ [0.1, 2.0]` Pa), matching the v1 physiological
  rationale.
