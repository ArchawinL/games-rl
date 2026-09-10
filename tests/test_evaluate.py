"""Milestone 4: evaluation-logic tests."""

import numpy as np
import pyspiel
import pytest
from open_spiel.python import rl_environment

from kuhn_nfsp.baselines import make_baseline
from kuhn_nfsp.config import load_config
from kuhn_nfsp.evaluate import _ci95, aggregate, evaluate_vs
from kuhn_nfsp.policies import load_avg_policy
from kuhn_nfsp.train import run_training

GAME = pyspiel.load_game("kuhn_poker")


def test_ci95_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.normal(size=50)
    big = rng.normal(size=50000)
    _, ci_small = _ci95(small)
    _, ci_big = _ci95(big)
    assert ci_big < ci_small
    assert _ci95([1.0])[1] == 0.0


def test_never_bet_vs_always_bet_is_deterministic_minus_one():
    # NeverBet always folds to a bet; AlwaysBet always bets. Whichever seat
    # NeverBet holds, it loses exactly 1 chip every hand -> mean -1, zero spread.
    returns = evaluate_vs(
        GAME,
        make_baseline("never_bet", GAME),
        make_baseline("always_bet", GAME),
        episodes=400,
        seed=1,
    )
    assert returns.shape == (400,)
    assert np.all(returns == -1.0)


def test_always_bet_beats_never_bet_by_one():
    returns = evaluate_vs(
        GAME,
        make_baseline("always_bet", GAME),
        make_baseline("never_bet", GAME),
        episodes=400,
        seed=2,
    )
    assert np.all(returns == 1.0)


def test_aggregate_across_seeds():
    fake = [
        {
            "run_dir": "experiments/a",
            "episodes": 10,
            "exploitability": 0.02,
            "nash_conv": 0.04,
            "vs": {n: {"mean_return": 0.1, "ci95": 0.01, "n": 10} for n in
                   ("random", "always_bet", "never_bet", "nash")},
        },
        {
            "run_dir": "experiments/b",
            "episodes": 10,
            "exploitability": 0.04,
            "nash_conv": 0.08,
            "vs": {n: {"mean_return": 0.3, "ci95": 0.01, "n": 10} for n in
                   ("random", "always_bet", "never_bet", "nash")},
        },
    ]
    agg = aggregate(fake)
    assert agg["n_seeds"] == 2
    assert agg["exploitability_mean"] == pytest.approx(0.03)
    assert agg["exploitability_std"] == pytest.approx(np.std([0.02, 0.04], ddof=1))
    assert agg["vs"]["random"]["mean_return"] == pytest.approx(0.2)


def test_load_avg_policy_roundtrip(tmp_path):
    cfg = load_config(
        None,
        [
            f"run_root={tmp_path}",
            "run_name=r",
            "num_episodes=300",
            "eval_every=300",
            "min_buffer_size_to_learn=32",
            "batch_size=16",
            "reservoir_buffer_capacity=2000",
            "replay_buffer_capacity=2000",
            "seed=5",
        ],
    )
    run_dir = run_training(cfg)
    env = rl_environment.Environment("kuhn_poker")
    policy = load_avg_policy(run_dir / "checkpoint_final.pt", env)

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        dist = policy.action_probabilities(state)
        assert set(dist) == set(state.legal_actions())
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        for a in state.legal_actions():
            walk(state.child(a))

    walk(GAME.new_initial_state())
