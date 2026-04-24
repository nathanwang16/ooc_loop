# Parametric geometry

The parametric chamber generator is defined in
`ooc_optimizer.geometry.generator` and produces both a fluid-domain STL
(for the BO inner loop) and a mold STL (for SLA printing).  The 2D BO mesh
itself is produced by `ooc_optimizer.geometry.topology_blockmesh`, not by
meshing the STL, because multi-block blockMesh is orders of magnitude
cheaper than STL-based meshing and gives exact inlet-patch naming.

## Continuous parameters

| Symbol    | Physical meaning                     | Default range | Units |
|-----------|--------------------------------------|---------------|-------|
| `W`       | Chamber width                        | 500 – 3000    | μm    |
| `d_p`     | Pillar diameter                      | 100 – 400     | μm    |
| `s_p`     | Pillar gap (centre-to-centre)        | 200 – 1000    | μm    |
| `theta`   | Inlet taper angle                    | 15 – 75       | deg   |
| `Q_total` | Total volumetric flow rate           | 5 – 200       | μL/min|
| `r_flow`  | `Q_drug / Q_total`                   | 0.1 – 0.9     | —     |
| `delta_W` | Inlet separation / `W` (`opposing` only) | 0.1 – 0.45 | —   |

See [`configs/default_config.yaml`](../../configs/default_config.yaml) for
the authoritative bound definitions.

## Discrete parameters

- **Pillar configuration**: `none`, `1x4`, `2x4`, `3x6`.
- **Chamber height**: 200 or 300 μm.
- **Inlet topology**: `opposing`, `same_side_Y`, `asymmetric_lumen`.

A full sweep is therefore `4 × 2 × 3 = 24` independent BO runs per target
profile.

## Inlet topologies

### `opposing`

Two short-side inlets at `x = 0`, centred at `y = W/2 ± δ`, separated by a
PDMS tongue of width `2δ`.  The drug inlet is below the centreline, the
medium inlet above.  `delta_W = δ / W` controls their separation.

### `same_side_Y`

Two arms meet upstream in a 90° Y-junction and the merged channel enters
the chamber as a single inlet face of width `W`.  In the BO surrogate the
inlet face is split into two half-height Dirichlet patches
(`inlet_drug` at `y < W/2`, `inlet_medium` at `y > W/2`) to model the
mixing that has already occurred upstream.

### `asymmetric_lumen`

Drug enters through a 60%-of-`L` lumen running along the `y = 0` wall;
medium enters at the full short edge `x = 0`.  This is the closest analogue
to the Ayuso-2020 chip layout.

## Patch-naming contract

All downstream code (meshing, solver, metrics, validation) expects exactly
these patch names:

- `inlet_drug`, `inlet_medium`
- `outlet`
- `walls`, `floor`
- `frontAndBack` (2D, type `empty`) **or** `floor` + `ceiling` (3D)

See `ooc_optimizer.geometry.topology_blockmesh` for the single source of
truth.

## API

::: ooc_optimizer.geometry.generator.generate_chip
::: ooc_optimizer.geometry.topology_blockmesh.generate_blockmesh_dict_v2
::: ooc_optimizer.geometry.topology_blockmesh.generate_blockmesh_dict_v2_3d
