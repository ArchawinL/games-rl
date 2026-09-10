"""Milestone 4: evaluate trained NFSP average policies.

    python -m kuhn_nfsp.evaluate --runs experiments/m3_s42 experiments/m3_s43 \
        --episodes 40000 --out results/summary.md

For each run it:
  * recomputes exploitability / nash_conv of the (exact) average policy, and
  * plays `episodes` hands (seats swapped at the midpoint) against each fixed
    baseline, reporting the trained policy's mean return with a 95% CI.
Results across runs (seeds) are aggregated into a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv

from kuhn_nfsp.baselines import make_baseline
from kuhn_nfsp.policies import load_avg_policy

OPPONENTS = ["random", "always_bet", "never_bet", "nash"]


def _ci95(samples) -> tuple[float, float]:
    x = np.asarray(samples, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, 0.0
    sem = x.std(ddof=1) / np.sqrt(len(x))
    return mean, float(1.96 * sem)


def _play_episode(game, policies, rng) -> float:
    """Play one hand; return player 0's terminal return."""
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=probs)))
        else:
            player = state.current_player()
            dist = policies[player].action_probabilities(state)
            actions = list(dist.keys())
            probs = np.array([dist[a] for a in actions], dtype=float)
            probs /= probs.sum()
            state.apply_action(int(rng.choice(actions, p=probs)))
    return state.returns()[0]


def evaluate_vs(game, trained, opponent, episodes: int, seed: int) -> np.ndarray:
    """Seat-swapped rollout; returns the trained policy's per-hand returns."""
    rng = np.random.default_rng(seed)
    half = episodes // 2
    out = np.empty(2 * half)
    for i in range(half):
        out[i] = _play_episode(game, {0: trained, 1: opponent}, rng)
    for i in range(half):
        out[half + i] = -_play_episode(game, {0: opponent, 1: trained}, rng)
    return out


def evaluate_run(
    run_dir: str | pathlib.Path,
    episodes: int,
    checkpoint: str = "final",
    seed: int = 0,
) -> dict:
    run_dir = pathlib.Path(run_dir)
    game = pyspiel.load_game("kuhn_poker")
    env = rl_environment.Environment("kuhn_poker")
    trained = load_avg_policy(run_dir / f"checkpoint_{checkpoint}.pt", env)

    result = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "episodes": episodes,
        "exploitability": float(exploitability(game, trained)),
        "nash_conv": float(nash_conv(game, trained)),
        "vs": {},
    }
    for name in OPPONENTS:
        returns = evaluate_vs(
            game, trained, make_baseline(name, game), episodes, seed
        )
        mean, ci = _ci95(returns)
        result["vs"][name] = {"mean_return": mean, "ci95": ci, "n": int(len(returns))}
    return result


def aggregate(results: list[dict]) -> dict:
    expl = [r["exploitability"] for r in results]
    agg = {
        "n_seeds": len(results),
        "episodes_per_seed": results[0]["episodes"] if results else 0,
        "exploitability_mean": float(np.mean(expl)),
        "exploitability_std": float(np.std(expl, ddof=1)) if len(expl) > 1 else 0.0,
        "exploitability_per_seed": expl,
        "vs": {},
        "per_seed": results,
    }
    for name in OPPONENTS:
        means = [r["vs"][name]["mean_return"] for r in results]
        agg["vs"][name] = {
            "mean_return": float(np.mean(means)),
            "std_across_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "per_seed": means,
        }
    return agg


def render_markdown(agg: dict) -> str:
    lines = [
        "# Kuhn NFSP — evaluation summary",
        "",
        f"- Seeds: **{agg['n_seeds']}**",
        f"- Episodes per seed per opponent: **{agg['episodes_per_seed']:,}** "
        "(seats swapped at midpoint)",
        f"- Exploitability of trained average policy: "
        f"**{agg['exploitability_mean']:.4f}** "
        f"(± {agg['exploitability_std']:.4f} across seeds); "
        f"per seed: {', '.join(f'{e:.4f}' for e in agg['exploitability_per_seed'])}",
        f"- Exact Nash exploitability = 0; uniform-random = 0.4583.",
        "",
        "## Mean return vs. fixed opponents (trained policy, seat-averaged)",
        "",
        "| Opponent | Mean return | Std across seeds | Per-seed means |",
        "|---|---|---|---|",
    ]
    labels = {
        "random": "UniformRandom",
        "always_bet": "AlwaysBet",
        "never_bet": "NeverBet",
        "nash": "AnalyticNash (α=1/6)",
    }
    for name in OPPONENTS:
        v = agg["vs"][name]
        per = ", ".join(f"{m:+.3f}" for m in v["per_seed"])
        lines.append(
            f"| {labels[name]} | **{v['mean_return']:+.3f}** | "
            f"{v['std_across_seeds']:.3f} | {per} |"
        )
    lines += [
        "",
        "Positive = trained policy wins chips per hand. Against AnalyticNash the "
        "trained policy cannot win (Nash is unexploitable); a value near 0 means "
        "the trained policy is itself close to optimal.",
        "",
        "## Per-seed detail",
        "",
        "| Seed run | Exploitability | vs Random | vs AlwaysBet | vs NeverBet | vs Nash |",
        "|---|---|---|---|---|---|",
    ]
    for r in agg["per_seed"]:
        vs = r["vs"]
        lines.append(
            f"| `{pathlib.Path(r['run_dir']).name}` | {r['exploitability']:.4f} | "
            + " | ".join(
                f"{vs[n]['mean_return']:+.3f} ± {vs[n]['ci95']:.3f}" for n in OPPONENTS
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Kuhn NFSP runs.")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="run directories (one per seed)"
    )
    parser.add_argument("--episodes", type=int, default=40000)
    parser.add_argument("--checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--seed", type=int, default=0, help="rollout RNG seed")
    parser.add_argument("--out", type=str, default="results/summary.md")
    args = parser.parse_args(argv)

    results = []
    for run in args.runs:
        print(f"evaluating {run} ...", flush=True)
        r = evaluate_run(run, args.episodes, args.checkpoint, args.seed)
        print(
            f"  exploitability={r['exploitability']:.4f}  "
            + "  ".join(f"{n}={r['vs'][n]['mean_return']:+.3f}" for n in OPPONENTS),
            flush=True,
        )
        results.append(r)

    agg = aggregate(results)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(agg))
    out_path.with_suffix(".json").write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
