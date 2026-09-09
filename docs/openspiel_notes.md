# OpenSpiel API notes (Kuhn poker)

Milestone 1 output. Written from exercising the API in `docs/api_spike.py`
against `open_spiel==2.0.2`. Every concrete number below is reproducible with:

```powershell
docker run --rm -v ${PWD}:/app kuhn-nfsp python docs/api_spike.py
```

The goal is to understand the interface well enough that, in later milestones,
the only thing being debugged is NFSP — not the framework.

---

## 1. `Game` vs `State`

OpenSpiel splits a game into two objects:

- **`Game`** — the immutable rulebook. Created once with
  `pyspiel.load_game("kuhn_poker")`. Exposes static facts: number of players,
  number of distinct actions, maximum game length, utility bounds, tensor
  shapes, and a `GameType` describing *kind* of game.
- **`State`** — a single mutable position in one playthrough. Created with
  `game.new_initial_state()` and advanced with `state.apply_action(a)`.
  `state.child(a)` returns an advanced **copy** (used for tree walks);
  `state.clone()` copies without advancing.

For Kuhn the `GameType` says: `SEQUENTIAL` dynamics, `EXPLICIT_STOCHASTIC`
chance, `IMPERFECT_INFORMATION`, `ZERO_SUM`. Those four properties are exactly
why NFSP (rather than plain self-play DQN) is the right tool.

Kuhn game-level facts:

| Property | Value |
|---|---|
| players | 2 |
| distinct actions | 2 — `0 = Pass/Check`, `1 = Bet/Call` |
| max game length | 3 moves |
| utility range | −2 … +2 |
| info-state tensor shape | `[11]` |

---

## 2. Node types

At any `State`, exactly one of these holds:

- **`state.is_chance_node()`** — the environment moves, not a player.
  `state.chance_outcomes()` returns `[(action, probability), ...]`. In Kuhn the
  first two moves are chance: deal a private card to P0 (3 equally likely
  outcomes), then to P1 (2 remaining, 0.5 each). Sampling a chance outcome is
  the caller's job.
- **`state.is_terminal()`** — the game is over; `state.returns()` is defined.
- otherwise it is a **decision node** for `state.current_player()` (an int
  player id `0` or `1`).

`state.history()` lists every action taken from the root, **chance actions
included**. Example terminal history `[2, 0, 0, 0]` = deal card 2 (King) to P0,
deal card 0 (Jack) to P1, P0 Pass, P1 Pass.

Special player ids: `pyspiel.PlayerId.CHANCE` (−1), `TERMINAL` (−4),
`SIMULTANEOUS` (−2, not used here).

---

## 3. Actions and legal actions

Actions are small non-negative ints. At a decision node:

- `state.legal_actions()` → list of currently legal action ints (Kuhn: always
  `[0, 1]`).
- `state.legal_actions_mask()` → 0/1 vector of length `num_distinct_actions`.
- `state.action_to_string(player, a)` → human label, e.g. `"Pass"`, `"Bet"`.

NFSP's networks output over the full action space; the illegal-action mask is
applied before sampling. In Kuhn nothing is ever masked, which keeps the first
implementation simple.

---

## 4. Information states (the core of imperfect information)

A player cannot see the opponent's card, so many distinct world histories look
identical to them. The set of histories a player cannot tell apart is an
**information state**. OpenSpiel gives two views, both from the acting player's
perspective:

- **`state.information_state_string(player)`** — a stable string key. Kuhn uses
  `"<card><betting-so-far>"`, e.g. `"0"`, `"2p"`, `"1pb"`. There are exactly
  **6 per player, 12 total**:

  ```
  P0: '0' '1' '2' '0pb' '1pb' '2pb'
  P1: '0p' '1p' '2p' '0b' '1b' '2b'
  ```

- **`state.information_state_tensor(player)`** — fixed-length `float` vector
  (length **11** for Kuhn) for feeding a network. Observed layout:

  | slice | meaning | example (`P1`, card Jack, after P0 Pass) |
  |---|---|---|
  | `[0:2]` | acting player one-hot | `[0, 1]` |
  | `[2:5]` | private card one-hot (J, Q, K) | `[1, 0, 0]` |
  | `[5:11]` | betting sequence, 2 bits/move, up to 3 moves | `[1, 0, 0, 0, 0, 0]` |

  (P0 holding the King before any bet is `[1,0, 0,0,1, 0,0,0,0,0,0]`.)

OpenSpiel games here have **perfect recall** — an information state encodes the
player's full observation history, so a policy indexed by information state is
well defined. This is what makes tabular best-response (and therefore
exploitability) computable.

`observation_string` / `observation_tensor` also exist but are a *smaller*,
possibly imperfect-recall view; NFSP here uses the **information** state.

---

## 5. Rewards and returns

- `state.returns()` — cumulative utility per player for the episode, valid at
  any node but only meaningful/terminal-accurate at a terminal. Zero-sum:
  `returns()[0] == -returns()[1]`.
- `state.rewards()` — per-step increment. Kuhn pays out only at the end, so
  `rewards()` is zero until the terminal, where it equals `returns()`.

Kuhn has **4 distinct terminal payoff vectors** — `±1` (someone folds to a bet,
or both check with the antes only) and `±2` (bet + call, then showdown) — across
**30 terminal histories**.

We use these returns **unshaped** as the RL reward (see the reward-design
section of the project README once written). Shaping would invalidate the
exploitability metric.

---

## 6. Policies and exploitability

- `open_spiel.python.policy.Policy` — maps an information state to an
  action→probability dict. `UniformRandomPolicy(game)` is the trivial baseline.
- `open_spiel.python.algorithms.exploitability`:
  - `nash_conv(game, policy)` — sum over players of how much each could gain by
    best-responding to the others. `0` iff the profile is a Nash equilibrium.
  - `exploitability(game, policy)` — `nash_conv / num_players`.

Reference values (our convergence yardsticks):

| Policy | exploitability | nash_conv |
|---|---|---|
| uniform random | **0.458333** | 0.916667 |
| exact Nash equilibrium | 0 | 0 |

The known Kuhn equilibrium value to P0 is **−1/18 ≈ −0.0556** (P0 is
structurally disadvantaged). Nash is a one-parameter family (α ∈ [0, 1/3]).

Sanity check from a 100k-episode **random-vs-random** rollout (seed 20260909):
mean return `P0 = +0.124`, `P1 = −0.124`. Under *uniform* random play P0 does
well; the −1/18 disadvantage only appears under *optimal* play.

---

## 7. The `rl_environment` wrapper (what NFSP consumes)

`open_spiel.python.rl_environment.Environment("kuhn_poker")` wraps the
`Game`/`State` machinery into a step/reset loop and **samples chance nodes
internally**, so the agent code only ever sees decision nodes.

- `env.reset()` / `env.step([action])` → a `TimeStep` with:
  - `observations["info_state"]` — list indexed by player, each the length-11
    information-state tensor.
  - `observations["legal_actions"]` — per-player legal action lists.
  - `observations["current_player"]` — whose turn (or a terminal marker).
  - `.rewards`, `.discounts`, `.last()`.
- `env.observation_spec()` → `{'info_state': (11,), 'legal_actions': (2,),
  'current_player': (), 'serialized_state': ()}`
- `env.action_spec()` → `{'num_actions': 2, 'min': 0, 'max': 1, 'dtype': int}`

OpenSpiel ships `open_spiel.python.pytorch.nfsp.NFSP`, an agent that consumes
exactly this `TimeStep` interface. Milestone 3 instantiates two of them (one per
seat), loops `reset`/`step` for self-play, and periodically wraps their
**average** policies to call `nash_conv` for the convergence curve.

---

## 8. Gotchas caught during the spike

- **Chance is the caller's responsibility** at the raw `State` level (not in
  `rl_environment`). Forgetting to check `is_chance_node()` before
  `legal_actions()` throws.
- **`information_state_string` needs a player argument** for imperfect-info
  games; calling it with none is ambiguous.
- Use the **information** state, not the **observation** state, for anything
  that must line up with `exploitability`.
- Bind-mount runs must use **PowerShell** (`docker run -v ${PWD}:/app ...`).
  Git Bash mangles the container path `/app` into a Windows path; prefix with
  `MSYS_NO_PATHCONV=1` if running from Git Bash.
