"""Fixed (non-learned) Kuhn poker policies used as evaluation opponents.

All of these implement ``open_spiel.python.policy.Policy`` and therefore work
directly with ``exploitability`` / ``best_response`` and with the play loop.

Kuhn actions:  ``0 = Pass`` (check or fold), ``1 = Bet`` (bet or call).
Information-state strings look like ``"<card><history>"`` where card is
``0=J, 1=Q, 2=K`` and history is one of ``"" / "p" / "b" / "pb"``:

    ""    P0's opening decision
    "p"   P0 checked, P1 to act
    "b"   P0 bet, P1 to act
    "pb"  P0 checked, P1 bet, P0 to act
"""

from __future__ import annotations

import pyspiel
from open_spiel.python.policy import Policy, UniformRandomPolicy

PASS, BET = 0, 1
_THIRD = 1.0 / 3.0


class RandomPolicy(UniformRandomPolicy):
    """OpenSpiel's uniform-random policy, re-exported under the project's name."""


class _KuhnBetProbPolicy(Policy):
    """Base class: a Kuhn policy fully described by P(bet) at each info state."""

    def __init__(self, game: pyspiel.Game):
        super().__init__(game, list(range(game.num_players())))

    def _bet_prob(self, card: str, history: str) -> float:
        raise NotImplementedError

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        legal = state.legal_actions(player_id)
        info_state = state.information_state_string(player_id)
        p_bet = min(1.0, max(0.0, self._bet_prob(info_state[0], info_state[1:])))
        full = {PASS: 1.0 - p_bet, BET: p_bet}
        return {a: full[a] for a in legal}


class AlwaysBetPolicy(_KuhnBetProbPolicy):
    """Bet / call at every decision. Maximally aggressive, trivially exploitable."""

    def _bet_prob(self, card: str, history: str) -> float:
        return 1.0


class NeverBetPolicy(_KuhnBetProbPolicy):
    """Check / fold at every decision. Maximally passive, trivially exploitable."""

    def _bet_prob(self, card: str, history: str) -> float:
        return 0.0


class AnalyticNashPolicy(_KuhnBetProbPolicy):
    """A closed-form Kuhn equilibrium.

    The equilibria form a one-parameter family in ``alpha`` in ``[0, 1/3]``.
    Player 0's play depends on ``alpha``; Player 1's equilibrium play is unique.
    Every ``alpha`` in range gives exploitability 0 (game value -1/18 to P0).
    """

    def __init__(self, game: pyspiel.Game, alpha: float = 1.0 / 6.0):
        if not 0.0 <= alpha <= _THIRD + 1e-12:
            raise ValueError(f"alpha must be in [0, 1/3], got {alpha}")
        super().__init__(game)
        self.alpha = float(alpha)

    def _bet_prob(self, card: str, history: str) -> float:
        a = self.alpha
        if history == "":                       # P0 opening
            return {"0": a, "1": 0.0, "2": 3.0 * a}[card]
        if history == "pb":                     # P0 facing P1's bet -> P(call)
            return {"0": 0.0, "1": a + _THIRD, "2": 1.0}[card]
        if history == "b":                      # P1 facing P0's bet -> P(call)
            return {"0": 0.0, "1": _THIRD, "2": 1.0}[card]
        if history == "p":                      # P1 after P0 checked -> P(bet)
            return {"0": _THIRD, "1": 0.0, "2": 1.0}[card]
        raise ValueError(f"unexpected Kuhn history segment: {history!r}")


#: name -> zero-arg-after-game constructor, for CLI / demo wiring.
BASELINES = {
    "random": RandomPolicy,
    "always_bet": AlwaysBetPolicy,
    "never_bet": NeverBetPolicy,
    "nash": AnalyticNashPolicy,
}


def make_baseline(name: str, game: pyspiel.Game) -> Policy:
    """Build a baseline policy by name (see ``BASELINES``)."""
    try:
        return BASELINES[name](game)
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r}; choices: {sorted(BASELINES)}"
        ) from None
