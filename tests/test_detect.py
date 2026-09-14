"""Milestone 10: sampler and detector sanity."""

import numpy as np
import pyspiel

from kuhn_nfsp.baselines import AnalyticNashPolicy, RandomPolicy
from kuhn_nfsp.detect import STATES, auroc, d1_score, d2_score, sample_counts, trajectories
from kuhn_nfsp.sandbag import to_policy
from kuhn_nfsp.train import policy_table

GAME = pyspiel.load_game("kuhn_poker")
STRUCTS = [trajectories(GAME, s) for s in (0, 1)]
NASH = policy_table(GAME, AnalyticNashPolicy(GAME))
ANTI = {s: 1 - p for s, p in NASH.items()}


def _rollout_counts(table, hands, rng):
    """Monte-Carlo reference: play hands vs uniform random, count agent decisions."""
    agent, opp = to_policy(GAME, table), RandomPolicy(GAME)
    counts = np.zeros((len(STATES), 2))
    for h in range(hands):
        seat = h % 2
        state = GAME.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probs)))
                continue
            player = state.current_player()
            dist = (agent if player == seat else opp).action_probabilities(state, player)
            action = int(rng.choice(list(dist), p=list(dist.values())))
            if player == seat:
                counts[STATES.index(state.information_state_string(player)), action] += 1
            state.apply_action(action)
    return counts


def test_trajectory_probabilities_sum_to_one():
    uniform = np.full(len(STATES), 0.5)
    for const, D in STRUCTS:
        probs = const * np.prod(uniform ** D[..., 1] * (1 - uniform) ** D[..., 0], axis=1)
        assert abs(probs.sum() - 1) < 1e-12


def test_sampler_matches_rollouts():
    rng = np.random.default_rng(0)
    exact = sample_counts(STRUCTS, NASH, 4000, 50, rng).mean(0)
    mc = _rollout_counts(NASH, 4000, rng)
    assert np.abs(exact - mc).max() < 60  # seeded; busiest cell sd ~26


def test_d1_null_vs_gap():
    rng = np.random.default_rng(1)
    a, b = (sample_counts(STRUCTS, NASH, 1000, 200, rng) for _ in range(2))
    c = sample_counts(STRUCTS, ANTI, 1000, 200, rng)
    null, gap = d1_score(a, b), d1_score(a, c)
    assert null.mean() < 20  # ~chi2 with <= 12 df
    assert gap.min() > null.max()


def test_d2_family_membership():
    rng = np.random.default_rng(2)
    family = np.array([[NASH[s] for s in STATES]])
    member = d2_score(sample_counts(STRUCTS, NASH, 1000, 200, rng), family, 1e-4)
    outsider = d2_score(sample_counts(STRUCTS, ANTI, 1000, 200, rng), family, 1e-4)
    assert member.mean() < 20
    assert outsider.min() > member.max()


def test_auroc_extremes():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
