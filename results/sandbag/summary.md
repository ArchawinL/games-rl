# Sandbagging study — results

Preregistration: [`prereg.md`](prereg.md). Locks: [`lock_table.md`](lock_table.md). Curves: [`detection_curves.png`](detection_curves.png). All rows: [`detection.csv`](detection.csv).

## Hypotheses (hidden 128, cap fit)

| | verdict | numbers |
|---|---|---|
| H1 | **pass** | max \|Δexpl0\| = 0.0004, max \|Δexpl1\| = 0.0009 |
| H2 | **pass** | leak h2=0.3453 vs h128=0.0014 (5/5 seeds); cap 100=0.0115 vs fit (5/5 seeds) |
| H3 | **pass** | mix@0.05 0.985, mix@0.10 0.992, mix@0.20 1.000, anti 1.000 |
| H4 | **pass** | max AUROC 0.524 (imitate@0.05, N=300) |
| H5 | **pass** | imitate@0.10 1.000, imitate@0.20 1.000, mix@0.10 1.000, mix@0.20 1.000, anti 1.000 |

## Detection (hidden 128, cap fit)

AUROC / TPR@5%FPR. D2 AUROC for `imitate`/`mix` is against `honest_weak` at the same level.

| sandbagger | D1 N=100 | D1 N=1000 | D1 N=10000 | D2 N=100 | D2 N=1000 | D2 N=10000 |
|---|---|---|---|---|---|---|
| `imitate@0.05` | 0.57 / 0.09 | 0.88 / 0.56 | 1.00 / 1.00 | 0.51 / 0.03 | 0.50 / 0.00 | 0.51 / 0.00 |
| `imitate@0.10` | 0.77 / 0.31 | 1.00 / 1.00 | 1.00 / 1.00 | 0.49 / 0.10 | 0.51 / 0.18 | 0.50 / 0.20 |
| `imitate@0.20` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 / 0.05 | 0.50 / 0.02 | 0.48 / 0.00 |
| `mix@0.05` | 0.67 / 0.13 | 0.96 / 0.82 | 1.00 / 1.00 | 0.72 / 0.18 | 0.98 / 0.42 | 1.00 / 0.86 |
| `mix@0.10` | 0.82 / 0.37 | 1.00 / 1.00 | 1.00 / 1.00 | 0.82 / 0.52 | 0.99 / 1.00 | 1.00 / 1.00 |
| `mix@0.20` | 0.96 / 0.81 | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 |
| `anti` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Held-out honest FPR at the calibrated threshold: D1 0.050, D2 0.051.

## Ablation (exploratory)

Mean over levels. D1 TPR at N=1000; D2 AUROC for `imitate` at N=10000.

| hidden | cap | D1 TPR (all positives) | D2 AUROC (imitate) |
|---|---|---|---|
| 128 | 100 | 0.910 | 0.504 |
| 128 | 1000 | 0.909 | 0.504 |
| 128 | fit | 0.911 | 0.497 |
| 8 | 100 | 0.696 | 0.662 |
| 8 | 1000 | 0.885 | 0.556 |
| 8 | fit | 0.891 | 0.556 |
| 2 | 100 | 0.427 | 0.613 |
| 2 | 1000 | 0.594 | 0.598 |
| 2 | fit | 0.616 | 0.596 |

## Deviations

None.
