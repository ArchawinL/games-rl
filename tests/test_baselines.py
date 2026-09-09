"""Milestone 2 acceptance tests for the fixed Kuhn policies."""

import pyspiel
import pytest
from open_spiel.python.algorithms.exploitability import exploitability

from kuhn_nfsp.baselines import (
    AlwaysBetPolicy,
    AnalyticNashPolicy,
    NeverBetPolicy,
    RandomPolicy,
    make_baseline,
)

GAME = pyspiel.load_game("kuhn_poker")
PASS, BET = 0, 1


def _decision_states(game):
    """Every non-chance, non-terminal state reachable in Kuhn."""
    out = []

    def rec(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                rec(state.child(action))
            return
        out.append(state.clone())
        for action in state.legal_actions():
            rec(state.child(action))

    rec(game.new_initial_state())
    return out


STATES = _decision_states(GAME)


def test_traversal_sees_every_information_state():
    keys = {s.information_state_string(s.current_player()) for s in STATES}
    assert len(keys) == 12  # 6 per player


@pytest.mark.parametrize(
    "policy",
    [
        RandomPolicy(GAME),
        AlwaysBetPolicy(GAME),
        NeverBetPolicy(GAME),
        AnalyticNashPolicy(GAME, 1.0 / 6.0),
    ],
    ids=["random", "always_bet", "never_bet", "nash"],
)
def test_valid_distribution_at_every_state(policy):
    for state in STATES:
        probs = policy.action_probabilities(state)
        legal = set(state.legal_actions())
        assert set(probs) == legal
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(-1e-12 <= p <= 1.0 + 1e-12 for p in probs.values())


def test_always_bet_and_never_bet_are_pure():
    for state in STATES:
        assert AlwaysBetPolicy(GAME).action_probabilities(state)[BET] == 1.0
        assert NeverBetPolicy(GAME).action_probabilities(state)[PASS] == 1.0


@pytest.mark.parametrize("alpha", [0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25, 1.0 / 3.0])
def test_analytic_nash_is_unexploitable(alpha):
    policy = AnalyticNashPolicy(GAME, alpha)
    assert exploitability(GAME, policy) < 1e-3


def test_analytic_nash_game_value_matches_theory():
    # exploitability = nash_conv / 2; nash_conv ~ 0 at equilibrium.
    assert exploitability(GAME, AnalyticNashPolicy(GAME, 1.0 / 6.0)) == pytest.approx(
        0.0, abs=1e-3
    )


@pytest.mark.parametrize("bad_alpha", [-0.01, 0.5, 1.0])
def test_analytic_nash_rejects_out_of_range_alpha(bad_alpha):
    with pytest.raises(ValueError):
        AnalyticNashPolicy(GAME, bad_alpha)


def test_baselines_beaten_by_best_response():
    # Sanity: the three trivial policies are genuinely exploitable.
    for name in ("random", "always_bet", "never_bet"):
        assert exploitability(GAME, make_baseline(name, GAME)) > 0.1


def test_make_baseline_unknown_name():
    with pytest.raises(ValueError):
        make_baseline("bluff_everything", GAME)
