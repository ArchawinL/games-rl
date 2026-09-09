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

### Milestone 3 — Training loop (1–2 days)
- `train.py`: config-driven (`configs/kuhn_nfsp.yaml` + CLI overrides), seeds
  Python/NumPy/framework RNGs, builds `rl_environment.Environment("kuhn_poker")`,
  runs two NFSP agents in self-play.
- Periodic evaluation every `eval_every` steps: wrap the agents' **average
  policies** via `policies.py`, compute `nash_conv`, append a row to
  `experiments/<run>/metrics.csv` (`step, nash_conv, loss_rl, loss_sl`).
- Checkpoint the best (lowest `nash_conv`) and the final policy; write
  `run_meta.json` (config, git SHA, seed, timestamp, wall-clock).
- Starting hyperparameters (from OpenSpiel's Kuhn NFSP example; tune if needed):
  hidden layers `[128]`, replay buffer `2e5`, reservoir buffer `2e6`,
  anticipatory param `0.1`, RL LR `0.01`, SL LR `0.01`, batch `128`,
  epsilon decay over first `~10%` of steps, training budget `3e5–1e6` episodes.
- **Done when:** a full run finishes and `metrics.csv` shows `nash_conv`
  decreasing to **≤ 0.05** within the configured budget.

### Milestone 4 — Evaluation (0.5–1 day)
- `evaluate.py`: load a checkpoint, play `M` episodes (default `M = 20000`,
  seats swapped each half) against each of: RandomPolicy, AlwaysBet, NeverBet,
  AnalyticNashPolicy. Report mean return ± 95% CI per opponent.
- Recompute final `exploitability` of the trained average policy.
- Run **≥ 3 seeds**; aggregate results into `results/summary.md`.
- **Done when:** `summary.md` contains the per-opponent table with CIs across
  seeds and the final exploitability figure.

### Milestone 5 — Analysis & plots (0.5 day)
- `plotting.py` produces, to `results/`:
  1. `exploitability_vs_steps.png` — mean ± band over seeds.
  2. `winrate_vs_random_vs_steps.png`.
  3. `final_return_by_opponent.png` — bar chart with CIs.
- Compare the learned policy's action probabilities per info state to the
  analytic Nash family; note which α it approaches.
- **Done when:** all three PNGs exist and are referenced in the README.

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

1. `docker build` + the full-config train command reaches **NashConv ≤ 0.05**
   within budget, across ≥ 3 seeds.
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
