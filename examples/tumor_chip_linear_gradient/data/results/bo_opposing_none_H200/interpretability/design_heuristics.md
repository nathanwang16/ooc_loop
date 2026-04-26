# Design heuristics

**Topology**: `opposing`  
**Pillar config**: `none`  
**Chamber height H**: `200 μm`  
**Target profile**: `{'kind': 'linear_gradient', 'axis': 'x', 'c_high': 1.0, 'c_low': 0.0}`

## Dominant parameters (global sensitivity)

- `Q_total`, `r_flow`

## Parameters that can be held loosely

- `theta`, `delta_W`

## Local sensitivity ranking (at the BO optimum)

| Parameter | |∂μ/∂x_norm| |
|---|---|
| `Q_total` | 0.4194 |
| `r_flow` | 0.2591 |
| `W` | 0.1012 |
| `delta_W` | 0.0587 |
| `theta` | 0.0004 |

## Tightest fabrication tolerances

| Parameter | −Δ (phys) | +Δ (phys) |
|---|---|---|
| `r_flow` | 0.04516 | 0.02276 |
| `W` | 611.7 | 12.86 |
| `Q_total` | 11.81 | 126.8 |
