# Table (f) — Surrogate-Quality Audit ($\Sigma S_T$)

Trustworthiness diagnostic for the post-hoc Sobol indices on each topology's
trained GP surrogate. A well-conditioned surrogate (no overfitting, no severe
input-correlation artefacts) yields $\Sigma S_T \approx 1$ to within Saltelli
sampling noise. Values above ~1.5 indicate overfit; the indices for that
topology should be reported with magnitude caveats.

| Topology | $H$ ($\mu$m) | Active dims | $\Sigma S_1$ | $\Sigma S_T$ | Verdict |
|---|---:|---:|---:|---:|---|
| Ladder | 300 | 2 | 0.989 | **1.013** | ✓ trustworthy |
| Ladder | 200 | 2 | 0.998 | **1.014** | ✓ trustworthy |
| Asymmetric lumen | 200 | 4 | 0.976 | **1.022** | ✓ trustworthy |
| Same-side Y | 200 | 4 | 0.898 | **1.064** | ✓ trustworthy |
| Opposing | 200 | 5 | 0.409 | **1.813** | ⚠ overfit — 39 % infeasibility rate compressed the GP into near-interpolation; magnitudes inflated |

**Reading.** Three of the five GPs are clean; `same_side_Y` is borderline-clean
(slight $S_1/S_T$ gap suggesting weak parameter interactions, consistent with
its $r_{\rm flow}$ + $W$ secondary structure); `opposing` is *not* trustworthy
and its parameter-ranking claims should carry an explicit caveat in any figure
where its bars appear.

**Why the audit matters.** All Sobol/local/tolerance results in §4 are claims
about the trained GP surrogate, not direct claims about the underlying CFD
response. $\Sigma S_T$ is the cheapest single-number self-check available
without rerunning CFD. Reporting it makes the surrogate-quality assumption
explicit and falsifiable.

**Source data.** `bo_<topology>_none_H<H>/interpretability/summary.json` —
fields `sobol.S1` and `sobol.ST`, summed across the active dimensions.
Sample size $n_{\rm sobol} = 1024$ Saltelli samples for the H=200 runs;
$n_{\rm sobol} = 4096$ for the H=300 ladder rerun.
