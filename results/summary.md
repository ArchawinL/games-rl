# Kuhn NFSP — evaluation summary

- Seeds: **3**
- Episodes per seed per opponent: **40,000** (seats swapped at midpoint)
- Exploitability of trained average policy: **0.0193** (± 0.0026 across seeds); per seed: 0.0178, 0.0178, 0.0223
- Exact Nash exploitability = 0; uniform-random = 0.4583.

## Mean return vs. fixed opponents (trained policy, seat-averaged)

| Opponent | Mean return | Std across seeds | Per-seed means |
|---|---|---|---|
| UniformRandom | **+0.133** | 0.005 | +0.130, +0.129, +0.139 |
| AlwaysBet | **+0.107** | 0.011 | +0.096, +0.105, +0.118 |
| NeverBet | **+0.185** | 0.004 | +0.188, +0.180, +0.186 |
| AnalyticNash (α=1/6) | **-0.012** | 0.001 | -0.011, -0.012, -0.014 |

Positive = trained policy wins chips per hand. Against AnalyticNash the trained policy cannot win (Nash is unexploitable); a value near 0 means the trained policy is itself close to optimal.

## Per-seed detail

| Seed run | Exploitability | vs Random | vs AlwaysBet | vs NeverBet | vs Nash |
|---|---|---|---|---|---|
| `m3_s42` | 0.0178 | +0.130 ± 0.014 | +0.096 ± 0.016 | +0.188 ± 0.010 | -0.011 ± 0.013 |
| `m3_s43` | 0.0178 | +0.129 ± 0.014 | +0.105 ± 0.016 | +0.180 ± 0.010 | -0.012 ± 0.013 |
| `m3_s44` | 0.0223 | +0.139 ± 0.014 | +0.118 ± 0.016 | +0.186 ± 0.010 | -0.014 ± 0.013 |
