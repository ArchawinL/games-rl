"""Milestone 1 API spike: exercise the OpenSpiel Game/State interface on Kuhn poker.

Run inside the container:
    docker run --rm -v ${PWD}:/app kuhn-nfsp python docs/api_spike.py

Every number printed here is summarised, in prose, in docs/openspiel_notes.md.
This script is a learning artifact, not part of the training pipeline.
"""

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
from open_spiel.python.policy import UniformRandomPolicy

SEED = 20260909


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def game_level(game):
    section("1. Game-level properties (pyspiel.Game)")
    t = game.get_type()
    print("short_name             :", t.short_name)
    print("dynamics               :", t.dynamics)          # SEQUENTIAL
    print("chance_mode            :", t.chance_mode)        # EXPLICIT_STOCHASTIC
    print("information            :", t.information)        # IMPERFECT_INFORMATION
    print("utility                :", t.utility)            # ZERO_SUM
    print("num_players            :", game.num_players())
    print("num_distinct_actions   :", game.num_distinct_actions())
    print("max_game_length        :", game.max_game_length())
    print("min / max utility      :", game.min_utility(), "/", game.max_utility())
    print("info_state_tensor_shape:", game.information_state_tensor_shape())


def walk_one_game(game):
    section("2. Walk one game: chance node -> player nodes -> terminal")
    rng = np.random.default_rng(SEED)
    state = game.new_initial_state()
    step = 0
    while not state.is_terminal():
        step += 1
        if state.is_chance_node():
            outcomes = state.chance_outcomes()             # [(action, prob), ...]
            actions, probs = zip(*outcomes)
            a = int(rng.choice(actions, p=probs))
            pretty = [(int(x), round(p, 3)) for x, p in outcomes]
            print(f"[{step}] CHANCE    outcomes={pretty} -> dealt action {a}")
            state.apply_action(a)
        else:
            p = state.current_player()
            legal = state.legal_actions()
            names = [state.action_to_string(p, x) for x in legal]
            print(f"[{step}] PLAYER {p}  legal={legal} {names}")
            print(f"          info_state_string(P{p}) = {state.information_state_string(p)!r}")
            tensor = np.asarray(state.information_state_tensor(p), dtype=int).tolist()
            print(f"          info_state_tensor(P{p}) = {tensor}")
            a = int(rng.choice(legal))
            print(f"          -> apply {a} ({state.action_to_string(p, a)})")
            state.apply_action(a)
    section("   terminal reached")
    print("history()  :", state.history())
    print("returns()  :", state.returns(), " (terminal-only rewards in Kuhn)")


def enumerate_info_states(game):
    section("3. Every information state (exhaustive tree traversal)")
    seen = {0: set(), 1: set()}
    terminal_returns = []

    def recurse(state):
        if state.is_terminal():
            terminal_returns.append(tuple(state.returns()))
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                recurse(state.child(a))
            return
        p = state.current_player()
        seen[p].add(state.information_state_string(p))
        for a in state.legal_actions():
            recurse(state.child(a))

    recurse(game.new_initial_state())
    for p in (0, 1):
        print(f"player {p}: {len(seen[p])} information states")
        for s in sorted(seen[p]):
            print(f"     {s!r}")
    print(f"\ndistinct terminal return vectors : {sorted(set(terminal_returns))}")
    print(f"total terminal histories         : {len(terminal_returns)}")


def random_vs_random(game, episodes=100_000):
    section(f"4. Random-vs-random rollout  ({episodes:,} episodes, seed {SEED})")
    rng = np.random.default_rng(SEED)
    totals = np.zeros(2)
    for _ in range(episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probs)))
            else:
                state.apply_action(int(rng.choice(state.legal_actions())))
        totals += state.returns()
    mean = totals / episodes
    print(f"mean return   P0 = {mean[0]:+.4f}    P1 = {mean[1]:+.4f}")
    print("zero-sum: the two means are negatives up to sampling noise")


def exploitability_reference(game):
    section("5. Exploitability / NashConv of the uniform-random policy")
    urp = UniformRandomPolicy(game)
    print("exploitability :", round(exploitability(game, urp), 6))
    print("nash_conv      :", round(nash_conv(game, urp), 6))
    print("known Kuhn game value to P0 at equilibrium : -1/18 =", round(-1 / 18, 6))
    print("exact Nash has exploitability 0 -> that is the NFSP convergence target")


def rl_env_view(game):
    section("6. rl_environment wrapper (the interface NFSP consumes)")
    env = rl_environment.Environment("kuhn_poker")
    ts = env.reset()
    print("num_players      :", env.num_players)
    print("observation_spec :", dict(env.observation_spec()))
    print("action_spec      :", dict(env.action_spec()))
    print("TimeStep.observations keys :", list(ts.observations.keys()))
    print("current_player   :", ts.observations["current_player"])
    print("len(info_state[0]):", len(ts.observations["info_state"][0]))


def main():
    game = pyspiel.load_game("kuhn_poker")
    game_level(game)
    walk_one_game(game)
    enumerate_info_states(game)
    random_vs_random(game)
    exploitability_reference(game)
    rl_env_view(game)


if __name__ == "__main__":
    main()
