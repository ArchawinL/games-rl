# Implementation Plan — NFSP Self-Play on Kuhn Poker (OpenSpiel)

**Deliverable #1:** a complete, self-contained deep-RL project. Train two
Neural Fictitious Self-Play (NFSP) agents by self-play on Kuhn poker via
OpenSpiel, then produce a measurable, reproducible result: exploitability
convergence over training, plus win rate versus baseline opponents.

This plan covers **only** Kuhn poker. No other games, no later roadmap phases.

---

## 1. Decisions (locked)

| Question | Choice | Rationale |
|---|---|---|
| Algorithm | **NFSP only** (deep RL self-play) | Canonical imperfect-information RL method in OpenSpiel; "pure RL" framing with no exact-solver crutch. |
| Runtime | **Docker** | OpenSpiel has no reliable native Windows path. All build/train/eval runs happen in a container. |
| Scope | **Kuhn poker only, polished** | One game, fully finished: training loop, curves, eval vs baselines, README, one-command repro. |
| Language | Python 3.11 | OpenSpiel's supported interpreter for current pip wheels. |
| Compute | CPU only | Kuhn NFSP networks are tiny (~1 hidden layer of 128); GPU is unnecessary. |

### Open items — resolved in Milestone 0
- **NFSP implementation:** ✅ **PyTorch** (`open_spiel.python.pytorch.nfsp`),
  imports cleanly once `dm-tree` is installed. No TensorFlow fallback needed.
- **Base image:** ✅ **`python:3.11-slim`** + `uv pip install`. The
  `open_spiel==2.0.2` manylinux wheel installs fine; only `libgomp1` is needed
  as an OS package. `torch==2.4.1+cpu` from the PyTorch CPU index.
- Exact versions pinned in `requirements.txt`; full resolved set in
  `requirements.lock`.

---

## 2. Background facts (used in tests and the analysis writeup)

- Kuhn poker: 2 players, 3-card deck {J, Q, K}, each antes 1, one betting round,
  actions = {pass/check, bet/call}. Terminal, zero-sum.
- Game value to Player 1 under optimal play: **−1/18 ≈ −0.0556** (Player 1 is
  structurally disadvantaged).
- Nash equilibrium is a **one-parameter family** (parameter α ∈ [0, 1/3]
  governing Player 1's bluff/bet frequency with the King). Exact Nash has
  **exploitability = 0**.
- `NashConv` = sum of both players' best-response gains against the current
  policy profile; it is the convergence metric we track. It requires a
  **tabular policy over all information states** — supplied by NFSP's
  **average (supervised) policy**, not its best-response (RL) head.

---

## 3. Reward design

Kuhn poker already defines the reward: the **native terminal chip return**
(±1 or ±2), zero-sum. The project uses it **unshaped**. Reward shaping would
break the game-theoretic interpretation and invalidate the exploitability
metric, so it is deliberately avoided.

The only reward-side engineering considered:
- Optional linear scaling of returns fed to the NFSP RL head (DQN-style) for
  gradient stability. Default: no scaling; document the value if one is used.
- Seat symmetry: agents train and are evaluated on **both seats** (P1 and P2),
  with returns seat-averaged, because the game is asymmetric.

The README's "Reward design" section states the above explicitly, with the
game's payoff table.

---

## 4. Repository layout

```
games-rl/
  CLAUDE.md                 # repo conventions for future sessions (Kuhn/Phase 1 only)
  IMPLEMENTATION_PLAN.md    # this file
  README.md                 # problem, method, raw `docker run` commands, results, limitations
  Dockerfile
  .dockerignore
  requirements.txt          # uv-installable direct deps (pinned)
  requirements.lock         # full resolved dependency set (direct + transitive)
  configs/
    kuhn_nfsp.yaml          # full training config
    kuhn_nfsp_smoke.yaml    # tiny budget for smoke test
  src/kuhn_nfsp/
    __init__.py
    config.py               # dataclasses + yaml loader + CLI override merge
    train.py                # self-play training entrypoint
    evaluate.py             # load checkpoint, eval vs baselines, recompute exploitability
    baselines.py            # random + heuristic + analytic-Nash policies
    policies.py             # NFSP average-policy wrapper (pyspiel.Policy-compatible)
    plotting.py             # curves + bar charts from logged metrics
  tests/
    test_baselines.py       # policy validity; analytic Nash exploitability ~ 0
    test_smoke_train.py     # 1–2k step run completes and writes metrics.csv
  docs/
    api_spike.py            # Milestone 1: runnable API exploration
    openspiel_notes.md      # Milestone 1: API concepts in our own words
  experiments/              # GITIGNORED: run dirs (checkpoints, metrics.csv, logs)
  results/                  # COMMITTED: final plots, summary.md, saved policy
```

No task runner for now (no Makefile / wrapper script). Everything is invoked
with raw `docker run`; wrapper scripts can be added later if the commands get
tedious.

### 4.1 Container commands (the only interface for now)

The image bakes in the source and sets `PYTHONPATH=/app/src`. Dev runs bind-mount
the working tree over `/app` so edits and `experiments/` / `results/` output land
on the host. From the repo root (PowerShell — use `$PWD`; on bash use `"$PWD"`):

```powershell
# Build
docker build -t kuhn-nfsp .

# Smoke check
docker run --rm kuhn-nfsp python -c "import pyspiel; g=pyspiel.load_game('kuhn_poker'); print(g.num_distinct_actions(), g.num_players())"

# Interactive shell
docker run --rm -it -v ${PWD}:/app kuhn-nfsp bash

# Train
docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.train --config configs/kuhn_nfsp.yaml

# Evaluate a run
docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.evaluate --run experiments/<run-dir>

# Regenerate plots
docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.plotting

# Tests
docker run --rm -v ${PWD}:/app kuhn-nfsp pytest -q
```

---

## 4.2 Testing philosophy (sidenote)

Targeted, not comprehensive. Test **our** logic, not OpenSpiel's. Worth a test:
`AnalyticNashPolicy` (hand-transcribed math — the `exploitability < 1e-3` check
proves it); `policies.py` NFSP average-policy wrapper (if it exposes the wrong
head, every exploitability number is wrong); `config.py` override merge; one
`train.py` smoke test (loop runs, `metrics.csv` written, checkpoint saved — a
CI guardrail, not a convergence check); one `evaluate.py` test on a deterministic
matchup. Skip: OpenSpiel's own behavior, training *quality* (that's the
exploitability curve's job), `plotting.py`. Target ~15–25 fast tests total plus
the one slow smoke test.

---

## 5. Milestones

Each milestone has a concrete, checkable output. Rough effort in parentheses
assumes familiarity with Python but not OpenSpiel.

### Milestone 0 — Docker environment ✅ DONE
- `Dockerfile` + `.dockerignore` written. Image: `python:3.11-slim` + `libgomp1`,
  `uv pip install --system -r requirements.txt`, copies `src/` + `configs/`,
  `ENV PYTHONPATH=/app/src`.
- `requirements.txt` pinned; `requirements.lock` captures the full resolved set.
- `docker build -t kuhn-nfsp .` succeeds; smoke check prints `2 2`.
- Verified in-container: `open_spiel==2.0.2`, `torch==2.4.1+cpu` (CUDA off),
  `open_spiel.python.pytorch.nfsp` imports, `rl_environment("kuhn_poker")` works
  (info-state size 11), `exploitability` module works, `kuhn_nfsp` importable
  via `PYTHONPATH`, `pytest` runs.
- **Bind-mount caveat:** `docker run -v ${PWD}:/app ...` works from **PowerShell**
  (the documented shell). Git Bash mangles the container path `/app`; prefix
  those with `MSYS_NO_PATHCONV=1` if used.
- **Reference number for Milestone 1:** exploitability of the uniform-random
  policy on Kuhn = **0.458333**.

### Milestone 1 — OpenSpiel API spike ✅ DONE
- `docs/api_spike.py` — runnable script exercising `Game`/`State`, chance nodes,
  legal actions, `information_state_string` / `_tensor`, `returns`/`rewards`,
  exhaustive info-state enumeration, random-vs-random rollout, exploitability,
  and the `rl_environment` wrapper.
- `docs/openspiel_notes.md` — the concepts in our own words, with reproducible
  numbers.
- Findings recorded: Kuhn has **12 information states** (6/player), **30 terminal
  histories**, **4 distinct payoff vectors** (±1, ±2); info-state tensor layout
  `[player one-hot 2 | card one-hot 3 | betting sequence 6]`. Uniform-random
  exploitability **0.458333** (nash_conv 0.916667). Random-vs-random mean return
  `P0 +0.124 / P1 −0.124` (seed 20260909). Equilibrium value to P0 is −1/18.
- **Done when:** `docs/openspiel_notes.md` explains Game/State, legal actions,
  information states, chance nodes, and returns in our own words, with the
  reference exploitability number recorded. ✅

### Milestone 2 — Baselines module ✅ DONE
- `src/kuhn_nfsp/baselines.py`: `RandomPolicy` (subclass of OpenSpiel's
  `UniformRandomPolicy`), `AlwaysBetPolicy`, `NeverBetPolicy`,
  `AnalyticNashPolicy(alpha)` (closed-form Kuhn equilibrium, one-parameter
  family α ∈ [0, 1/3]; P1's play is unique), plus a `make_baseline(name, game)`
  factory + `BASELINES` registry for CLI/demo wiring.
- `tests/test_baselines.py`: **17 tests pass** — valid distributions at all 12
  info states, pure policies are pure, `AnalyticNashPolicy` exploitability < 1e-3
  across the whole α family, out-of-range α rejected, trivial policies confirmed
  exploitable.
- Reference exploitability (opponents for Milestone 4):

  | policy | exploitability | nash_conv |
  |---|---|---|
  | `never_bet` | 1.000000 | 2.000000 |
  | `random` | 0.458333 | 0.916667 |
  | `always_bet` | 0.333333 | 0.666667 |
  | `nash` (any α) | ~2.8e-17 | ~0 |

- **Done when:** `tests/test_baselines.py` passes; every policy returns a valid
  distribution over legal actions for every info state; and
  `exploitability(AnalyticNashPolicy(1/6)) < 1e-3`. ✅

### Milestone 3 — Training loop ✅ DONE
- `src/kuhn_nfsp/config.py` — frozen `TrainConfig` dataclass, YAML loader,
  `--set KEY=VALUE` overrides with type coercion. `tests/test_config.py` (8).
- `src/kuhn_nfsp/policies.py` — `NFSPAveragePolicy`, wraps the **average**
  (supervised) head of the NFSP agents as an OpenSpiel `Policy` for `nash_conv`.
  Mirrors `NFSPPolicies` from the stock `nfsp_kuhn_pytorch.py`.
- `src/kuhn_nfsp/train.py` — two `NFSP` agents, self-play over `rl_environment`,
  `nash_conv` eval every `eval_every` episodes → `experiments/<run>/metrics.csv`
  (`episode, agent_steps, elapsed_s, nash_conv, exploitability, sl/rl losses`),
  `run_meta.json` (config + git SHA + seed + lib versions), `checkpoint_best.pt`
  / `checkpoint_final.pt` (own format — OpenSpiel's `NFSP.save`/`restore` are
  broken in 2.0.2, key mismatch), `summary.json`. `tests/test_smoke_train.py` (3).
- `configs/kuhn_nfsp.yaml` (3e6 episodes) + `configs/kuhn_nfsp_smoke.yaml`.
- **Reproduction check:** our curve matches OpenSpiel's stock `nfsp_kuhn_pytorch.py`
  within run-to-run noise at every eval (ep20k 0.26 vs 0.27; ep40k 0.22 vs 0.22;
  ep80k 0.15 vs 0.16). NFSP on Kuhn plateaus near **0.14 exploitability by
  ~100k episodes** and then grinds down slowly — the stock example's 3e6-episode
  default is not optional. Also fixed: `epsilon_decay_duration` must track the
  episode budget or the best-response head keeps ~6% exploration forever and the
  average policy stalls higher.
- **Revised target:** **exploitability ≤ 0.05** (nash_conv ≤ 0.10) — the standard
  "converged" bar for NFSP; the earlier `nash_conv ≤ 0.05` (expl ≤ 0.025) is
  single-seed-variance-sensitive at 3e6 episodes.
- **Production run `experiments/m3_s42` (seed 42, 3e6 episodes, 44 min, CPU):**
  final exploitability **0.0178** (nash_conv 0.0356), best **0.0140**
  (nash_conv 0.0279). Beats both the revised bar and the original strict one.
  Curve: 0.40 → 0.14 (plateau, ~ep 100k) → 0.05 (~ep 1.37M) → ~0.018 (ep 3M).
  Artifacts: `metrics.csv` (300 evals), `checkpoint_best.pt`,
  `checkpoint_final.pt`, `run_meta.json` (git SHA `2054cdd`), `summary.json`.
- **Done when:** the 3e6-episode run finishes and `metrics.csv` shows
  exploitability reaching **≤ 0.05**; checkpoints + `run_meta.json` +
  `summary.json` written. ✅

### Milestone 4 — Evaluation ✅ DONE
- `src/kuhn_nfsp/policies.py::load_avg_policy` — rebuild an `NFSPAveragePolicy`
  from a checkpoint (reconstruct bare NFSP agents, load the saved avg-net weights).
- `src/kuhn_nfsp/evaluate.py` — per run: recompute `exploitability` / `nash_conv`
  of the average policy; play `--episodes` hands (default 40000, seats swapped at
  the midpoint) vs `random` / `always_bet` / `never_bet` / `nash`; report the
  trained policy's mean return ± 95% CI. Aggregates runs into `results/summary.md`
  + `summary.json`.
- `tests/test_evaluate.py` (5) — CI shrinks with n; `never_bet` vs `always_bet`
  is a deterministic −1.0; multi-seed aggregation; checkpoint round-trip. Suite
  **35 passing**.
- **3-seed result (`m3_s42/43/44`, 40k hands each, `results/summary.md`):**

  | | value |
  |---|---|
  | exploitability of trained avg policy | **0.0193 ± 0.0026** (0.0178, 0.0178, 0.0223) |
  | mean return vs UniformRandom | **+0.133 ± 0.005** |
  | mean return vs AlwaysBet | **+0.107 ± 0.011** |
  | mean return vs NeverBet | **+0.185 ± 0.004** |
  | mean return vs AnalyticNash (α=1/6) | **−0.012 ± 0.001** (≈ 0 → near-optimal) |

- **Done when:** `summary.md` has the per-opponent table with CIs across ≥ 3
  seeds and the final exploitability figure. ✅

### Milestone 5 — Analysis & plots ✅ DONE
- `src/kuhn_nfsp/plotting.py` produces, to `results/`:
  1. `exploitability_vs_steps.png` — 3-seed mean + min/max band, log-y, with the
     0.05 target line. Shows the plateau (~ep 0.3–0.7M) then the descent to ~0.02.
  2. `final_return_by_opponent.png` — bar chart, mean return ± std-across-seeds.
  3. `policy_vs_nash.md` — P(bet) at all 12 info states, trained vs analytic Nash.
- **`winrate_vs_random_vs_steps.png` cut** — `train.py` only logs `nash_conv` per
  eval, not win-rate-vs-random; that plot would need re-running 3× 3e6-episode
  training to log an extra column. The exploitability curve is the convergence
  story; not worth ~6 h of retraining for a secondary plot.
- **Policy analysis:** the trained avg policy tracks the Nash family with implied
  **α ≈ 0.158** (≈ midpoint of [0, 1/3]). Opening bets: Jack 0.158 vs 0.158,
  King 0.474 vs 0.475 (= 3α); Jack-facing-check bluff 0.334 vs 0.333; Queen
  bluff-catch 0.532 vs 0.492 (= α + 1/3). Residual gaps at `0b`/`0pb` (~0.03 vs 0)
  match the 0.018 measured exploitability.
- **Done when:** the PNGs exist and are referenced in the README. ✅ (referenced
  in Milestone 6)

### Milestone 6 — Package & document (0.5 day)
- `README.md`: problem statement, method (NFSP, one paragraph), how to run
  (the raw `docker run` commands from section 4.1 only), results (embedded PNGs
  + summary table), the reward-design section, limitations, and a verified repro
  sequence from a clean clone.
- Commit `results/` artifacts (plots, `summary.md`, saved policy file).
- Optional stretch: GitHub Actions running the test + smoke-train `docker run`
  commands.
- **Done when:** from a clean clone, `docker build -t kuhn-nfsp .` then the
  `pytest` and `--config configs/kuhn_nfsp_smoke.yaml` train commands run green,
  and the train / evaluate / plotting commands reproduce the committed results
  shape.

---

## 6. Definition of Done (whole deliverable)

1. `docker build` + the full-config train command reaches **exploitability ≤ 0.05**
   (nash_conv ≤ 0.10) within budget, across ≥ 3 seeds.
2. Trained average policy beats `RandomPolicy` by a **statistically significant**
   seat-averaged margin (mean return + 95% CI reported).
3. `exploitability_vs_steps.png` shows a broadly monotone decrease, plotted over
   ≥ 3 seeds.
4. One-command repro is documented and verified from a clean clone.
5. README explains the reward-design decision (native zero-sum terminal return,
   no shaping) and the evaluation methodology.
6. `docker run --rm -v ${PWD}:/app kuhn-nfsp pytest -q` passes.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `open_spiel` wheel won't install/import on `slim` image | Fall back to `python:3.11` full image or upstream OpenSpiel Dockerfile (Milestone 0). |
| PyTorch NFSP module missing/broken in pinned release | Fall back to TF NFSP; pin compatible `tensorflow` (Milestone 0). |
| Exploitability computed from the wrong NFSP head | `policies.py` wraps the **average/SL** policy only; unit-test that wrapper on a known policy. |
| High training variance between seeds | Fixed seed list, ≥ 3 seeds, report CIs; keep raw `metrics.csv` per run. |
| Non-reproducible runs | `run_meta.json` records config + git SHA + seed; configs are version-controlled; all runs are the fixed `docker run` commands in section 4.1. |
| Scope creep (Leduc, other games, later phases) | Out of scope by decision; this plan and `CLAUDE.md` are Kuhn-only. |

---

## 8. First actions

1. Create `experiments/` gitignore entry. *(done)*
2. Milestone 0: write `Dockerfile` + `.dockerignore`; `docker build -t kuhn-nfsp .`;
   run the smoke check; resolve the two open items and pin exact versions in
   `requirements.txt`.
