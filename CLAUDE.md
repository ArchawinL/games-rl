# CLAUDE.md

## What this repo is

A self-contained deep-RL project: train two **NFSP** (Neural Fictitious
Self-Play) agents by self-play on **Kuhn poker** via **OpenSpiel**, and produce
a reproducible result — exploitability (NashConv) convergence over training,
plus win rate versus baseline opponents.

Scope is **Kuhn poker only**. Do not add other games or unrelated features.
See `IMPLEMENTATION_PLAN.md` for the full plan and milestones.

## Environment

- **Everything runs in Docker.** OpenSpiel has no reliable native Windows path.
  Do not assume the host Python has `pyspiel` / `open_spiel`.
- No task runner for now — invoke everything with raw `docker run` (PowerShell,
  from the repo root). Image tag: `kuhn-nfsp`.
  - Build: `docker build -t kuhn-nfsp .`
  - Shell: `docker run --rm -it -v ${PWD}:/app kuhn-nfsp bash`
  - Train: `docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.train --config configs/kuhn_nfsp.yaml`
  - Eval: `docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.evaluate --run experiments/<run-dir>`
  - Plots: `docker run --rm -v ${PWD}:/app kuhn-nfsp python -m kuhn_nfsp.plotting`
  - Tests: `docker run --rm -v ${PWD}:/app kuhn-nfsp pytest -q`
- Dependencies live in `requirements.txt`, installed with `uv` inside the image.
  The image sets `PYTHONPATH=/app/src`.

## Conventions

- Runs are **config-driven** (`configs/*.yaml`) with CLI overrides; never
  hard-code hyperparameters in `train.py`.
- **Seed everything** (Python, NumPy, torch/TF). Each run writes
  `run_meta.json` with config + git SHA + seed.
- Outputs: run dirs go to `experiments/` (**gitignored**). Final committed
  artifacts (plots, `summary.md`, saved policy) go to `results/`.
- Evaluate on **both seats** (Kuhn is asymmetric); report seat-averaged returns
  with 95% CIs over ≥ 3 seeds.

## Gotchas

- Compute exploitability from NFSP's **average (supervised) policy**, never the
  best-response (RL) head. The `src/kuhn_nfsp/policies.py` wrapper enforces this.
- Rewards are the **native zero-sum terminal chip return, unshaped**. Shaping
  would invalidate the exploitability metric.
- Reference numbers: Kuhn game value to Player 1 under optimal play is
  **−1/18 ≈ −0.0556**; exact Nash exploitability is **0**; target for the
  trained agent is **NashConv ≤ 0.05**.
- Prefer PyTorch NFSP (`open_spiel.python.pytorch.nfsp`); fall back to the TF
  implementation only if that module is unavailable in the pinned release.
