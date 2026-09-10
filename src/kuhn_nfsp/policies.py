"""Expose NFSP's *average* (supervised) head as an OpenSpiel ``policy.Policy``.

NFSP carries two policies per player: a best-response (RL/DQN) head and an
average head. Only the **average** head converges toward Nash, so exploitability
and ``nash_conv`` must be computed against this wrapper, never the RL head.

Pattern mirrors ``NFSPPolicies`` from OpenSpiel's ``examples/kuhn_nfsp.py``.
"""

from __future__ import annotations

import pathlib

import torch
from open_spiel.python import policy as policy_lib
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.nfsp import MODE, NFSP


class NFSPAveragePolicy(policy_lib.Policy):
    """Wraps a list of live NFSP agents (one per player) as a joint policy."""

    def __init__(self, env, nfsp_agents):
        super().__init__(env.game, list(range(len(nfsp_agents))))
        self._agents = list(nfsp_agents)
        n = len(self._agents)
        self._obs = {"info_state": [None] * n, "legal_actions": [None] * n}

    def action_probabilities(self, state, player_id=None):
        cur = state.current_player() if player_id is None else player_id
        legal_actions = state.legal_actions(cur)

        self._obs["current_player"] = cur
        self._obs["info_state"][cur] = state.information_state_tensor(cur)
        self._obs["legal_actions"][cur] = legal_actions

        step = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None
        )
        with self._agents[cur].temp_mode_as(MODE.AVERAGE_POLICY):
            probs = self._agents[cur].step(step, is_evaluation=True).probs

        return {action: float(probs[action]) for action in legal_actions}


def load_avg_policy(checkpoint_path: str | pathlib.Path, env) -> NFSPAveragePolicy:
    """Rebuild an `NFSPAveragePolicy` from a `train.save_checkpoint` file.

    Reconstructs bare NFSP agents (the inner DQN is created but unused) and loads
    the saved average-policy network weights into each.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if ckpt.get("format") != "kuhn_nfsp.avg_policy.v1":
        raise ValueError(f"unexpected checkpoint format: {ckpt.get('format')!r}")

    cfg = ckpt["config"]
    agents = []
    for pid in range(2):
        agent = NFSP(
            pid,
            ckpt["info_state_size"],
            ckpt["num_actions"],
            hidden_layers_sizes=list(ckpt["hidden_layers"]),
            reservoir_buffer_capacity=cfg["reservoir_buffer_capacity"],
            anticipatory_param=cfg["anticipatory_param"],
            seed=cfg["seed"] + 17 * pid,
        )
        agent._avg_network.load_state_dict(ckpt["avg_network_state_dicts"][pid])
        agent._avg_network.eval()
        agents.append(agent)

    return NFSPAveragePolicy(env, agents)
