"""Milestone 5: figures + a trained-policy-vs-analytic-Nash table.

    python -m kuhn_nfsp.plotting --runs experiments/m3_s42 experiments/m3_s43 \
        experiments/m3_s44 --summary results/summary.json --outdir results

Writes to --outdir:
    exploitability_vs_steps.png     mean + seed min/max band, from each metrics.csv
    final_return_by_opponent.png    bar chart from summary.json
    policy_vs_nash.md               P(bet) per info state, trained vs analytic Nash
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import rl_environment  # noqa: E402

from kuhn_nfsp.baselines import AnalyticNashPolicy  # noqa: E402
from kuhn_nfsp.policies import load_avg_policy  # noqa: E402

BET = 1
_THIRD = 1.0 / 3.0


def _load_curve(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((pathlib.Path(run_dir) / "metrics.csv").open()))
    episode = np.array([int(r["episode"]) for r in rows])
    expl = np.array([float(r["exploitability"]) for r in rows])
    return episode, expl


def plot_exploitability(run_dirs, out_path, target: float = 0.05) -> None:
    curves = [_load_curve(d) for d in run_dirs]
    n = min(len(e) for _, e in curves)
    episode = curves[0][0][:n]
    stack = np.vstack([e[:n] for _, e in curves])
    mean = stack.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(episode, stack.min(0), stack.max(0), alpha=0.2, label="seed min–max")
    ax.plot(episode, mean, lw=2, label=f"mean of {len(run_dirs)} seeds")
    ax.axhline(target, ls="--", color="grey", label=f"target {target}")
    ax.set_yscale("log")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("exploitability (log scale)")
    ax.set_title("Kuhn NFSP — average-policy exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_final_returns(summary_path, out_path) -> None:
    agg = json.loads(pathlib.Path(summary_path).read_text())
    names = ["random", "always_bet", "never_bet", "nash"]
    labels = ["vs Random", "vs AlwaysBet", "vs NeverBet", "vs Nash"]
    means = [agg["vs"][k]["mean_return"] for k in names]
    errs = [agg["vs"][k]["std_across_seeds"] for k in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=errs, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("trained-policy mean return / hand")
    ax.set_title(f"Trained NFSP vs fixed opponents ({agg['n_seeds']} seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _info_state_examples(game) -> dict:
    """One reachable State per information state (12 for Kuhn)."""
    seen: dict = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        key = state.information_state_string(state.current_player())
        seen.setdefault(key, state.clone())
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return seen


def policy_table(run_dir, out_path) -> None:
    game = pyspiel.load_game("kuhn_poker")
    env = rl_environment.Environment("kuhn_poker")
    trained = load_avg_policy(pathlib.Path(run_dir) / "checkpoint_final.pt", env)
    states = _info_state_examples(game)

    p_bet = {k: trained.action_probabilities(s).get(BET, 0.0) for k, s in states.items()}
    alpha = p_bet["0"]  # analytic Nash: P(P0 bets a Jack on the opening) = alpha
    nash = AnalyticNashPolicy(game, min(max(alpha, 0.0), _THIRD))
    nash_bet = {k: nash.action_probabilities(s).get(BET, 0.0) for k, s in states.items()}

    lines = [
        f"# Trained avg-policy vs analytic Nash (`{pathlib.Path(run_dir).name}`)",
        "",
        f"Implied α (from P0 bet-with-Jack) ≈ **{alpha:.3f}** — Nash family is α ∈ [0, 1/3].",
        "",
        "| info state | P(bet) trained | P(bet) Nash(α) |",
        "|---|---|---|",
    ]
    for k in sorted(states):
        lines.append(f"| `{k}` | {p_bet[k]:.3f} | {nash_bet[k]:.3f} |")
    pathlib.Path(out_path).write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kuhn NFSP figures + policy table.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--summary", default="results/summary.json")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_exploitability(args.runs, out / "exploitability_vs_steps.png")
    plot_final_returns(args.summary, out / "final_return_by_opponent.png")
    policy_table(args.runs[0], out / "policy_vs_nash.md")
    print("wrote:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
