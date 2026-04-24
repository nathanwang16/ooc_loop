# Design heuristics

**Topology**: `opposing`  
**Pillar config**: `none`  
**Chamber height H**: `200 μm`  
**Target profile**: `{'kind': 'linear_gradient', 'axis': 'x', 'c_high': 1.0, 'c_low': 0.0}`

## Dominant parameters (global sensitivity)

- `W`, `theta`, `r_flow`

## Parameters that can be held loosely

- `delta_W`

## Local sensitivity ranking (at the BO optimum)

| Parameter | |∂μ/∂x_norm| |
|---|---|
| `theta` | 0.1535 |
| `r_flow` | 0.1441 |
| `W` | 0.1276 |
| `Q_total` | 0.0150 |
| `delta_W` | 0.0004 |

## Tightest fabrication tolerances

| Parameter | −Δ (phys) | +Δ (phys) |
|---|---|---|
| `W` | 1205 | 595.1 |
| `theta` | 14.59 | 45.41 |
| `Q_total` | 74.44 | 120.6 |
