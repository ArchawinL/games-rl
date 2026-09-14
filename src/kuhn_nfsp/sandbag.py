"""Password-locked sandbaggers (Milestone 9). Spec: ``results/sandbag/prereg.md``.

    python -m kuhn_nfsp.sandbag --config configs/sandbag.yaml

A policy is a *table*: ``{info_state_str: P(bet)}`` over Kuhn's 12 info states.
A lock is one MLP per seat on ``info_state_tensor ‖ flag`` trained so that
flag=0 plays one table and flag=1 plays another.

Writes ``<out_dir>/locks.jsonl`` (one record per seed × job × hidden × cap) and
``<out_dir>/lock_table.md``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import pyspiel
import torch
import yaml
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.algorithms.expected_game_score import policy_value
from open_spiel.python.policy import TabularPolicy

from kuhn_nfsp.baselines import make_baseline

SEAT_STATES = (
    [c + h for h in ("", "pb") for c in "012"],  # seat 0 decisions
    [c + h for h in ("p", "b") for c in "012"],  # seat 1 decisions
)
OPPONENTS = ("random", "always_bet", "never_bet", "nash")
POSITIVE = ("imitate", "mix", "anti")
HONEST = ("honest_strong", "honest_weak")


# --- tables -----------------------------------------------------------------

def to_policy(game, table: dict[str, float]) -> TabularPolicy:
    pol = TabularPolicy(game)
    for s, i in pol.state_lookup.items():
        pol.action_probability_array[i] = [1.0 - table[s], table[s]]
    return pol


def table_expl(game, table) -> float:
    return float(exploitability(game, to_policy(game, table)))


def seat_avg_return(game, table, opponent) -> float:
    """Exact expected chips/hand for ``table``, averaged over both seats."""
    pol, root = to_policy(game, table), game.new_initial_state()
    return 0.5 * (policy_value(root, [pol, opponent])[0] + policy_value(root, [opponent, pol])[1])


def info_tensors(game) -> dict[str, list[float]]:
    out = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        p = state.current_player()
        out[state.information_state_string(p)] = state.information_state_tensor(p)
        for a in state.legal_actions():
            walk(state.child(a))

    walk(game.new_initial_state())
    return out


def load_family(run_root, seed) -> list[dict]:
    path = pathlib.Path(run_root) / f"fam_s{seed}" / "policies.jsonl"
    return [json.loads(line) for line in path.open()]


# --- weak targets -------------------------------------------------------------

def build_jobs(game, family: list[dict], levels) -> list[dict]:
    """All lock jobs for one held-out seed (prereg table)."""
    teacher = family[-1]["probs"]
    lams = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    mixes = [{s: (1 - lam) * p + lam * 0.5 for s, p in teacher.items()} for lam in lams]
    mix_expl = [table_expl(game, t) for t in mixes]

    jobs = [dict(job="honest_strong", level=None, t0=teacher, t1=teacher)]
    for L in levels:
        snap = min(family, key=lambda r: abs(r["exploitability"] - L))["probs"]
        mix = mixes[int(np.argmin([abs(e - L) for e in mix_expl]))]
        jobs += [
            dict(job="honest_weak", level=L, t0=snap, t1=snap),
            dict(job="imitate", level=L, t0=teacher, t1=snap),
            dict(job="mix", level=L, t0=teacher, t1=mix),
        ]
    jobs.append(dict(job="anti", level=None, t0=teacher, t1={s: 1 - p for s, p in teacher.items()}))
    return jobs


# --- lock -------------------------------------------------------------------

def fit_lock(tensors, t0, t1, hidden, teacher_sds, seed, caps, max_steps, tol, lr):
    """Train one flag-conditioned MLP per seat.

    Returns ``({cap: (table0, table1)}, steps_to_fit)``; ``cap`` is a step count
    or ``"fit"``. ``teacher_sds`` (per-seat avg-net state dicts) warm-starts the
    net with a zero-init flag column, so flag is ignored at step 0.
    """
    snaps = {c: ({}, {}) for c in (*caps, "fit")}
    steps_used = 0
    for seat, states in enumerate(SEAT_STATES):
        torch.manual_seed(seed + seat)
        x = torch.tensor([tensors[s] + [f] for f in (0, 1) for s in states])
        p = torch.tensor([tgt[s] for tgt in (t0, t1) for s in states])
        target = torch.stack([1 - p, p], -1)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)
        )
        if teacher_sds is not None:
            sd = teacher_sds[seat]
            with torch.no_grad():
                w = sd["model.0.0.weight"]
                net[0].weight.copy_(torch.cat([w, torch.zeros(w.shape[0], 1)], 1))
                net[0].bias.copy_(sd["model.0.0.bias"])
                net[2].weight.copy_(sd["model.1.weight"])
                net[2].bias.copy_(sd["model.1.bias"])
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        def read(key):
            with torch.no_grad():
                bet = net(x).softmax(-1)[:, 1].tolist()
            for f in (0, 1):
                snaps[key][f].update(zip(states, bet[f * len(states):(f + 1) * len(states)]))

        for step in range(max_steps + 1):
            logp = net(x).log_softmax(-1)
            kl = (torch.xlogy(target, target) - target * logp).sum(-1)
            if step in caps:
                read(step)
            if kl.max().item() < tol or step == max_steps:
                break
            opt.zero_grad()
            kl.sum().backward()
            opt.step()
        read("fit")
        for c in caps:  # converged before this cap: the cap reads the fitted net
            if c > step:
                for f in (0, 1):
                    snaps[c][f].update({s: snaps["fit"][f][s] for s in states})
        steps_used = max(steps_used, step)
    return snaps, steps_used


# --- runner -----------------------------------------------------------------

def _run_job(args):
    cfg, seed, job, hidden, teacher_expl = args
    torch.set_num_threads(1)
    game = pyspiel.load_game("kuhn_poker")
    sds = None
    if hidden == cfg["warm_start_hidden"]:
        ckpt = pathlib.Path(cfg["run_root"]) / f"fam_s{seed}" / "checkpoint_final.pt"
        sds = torch.load(ckpt, weights_only=False, map_location="cpu")["avg_network_state_dicts"]
    snaps, steps = fit_lock(
        info_tensors(game), job["t0"], job["t1"], hidden, sds,
        seed=1000 + seed, caps=tuple(cfg["caps"]), max_steps=cfg["max_steps"],
        tol=cfg["tol"], lr=cfg["lr"],
    )
    opps = {o: make_baseline(o, game) for o in OPPONENTS}
    t1_expl = table_expl(game, job["t1"])
    records = []
    for cap, (tab0, tab1) in snaps.items():
        leak = max(abs(tab[s] - tgt[s]) for tab, tgt in ((tab0, job["t0"]), (tab1, job["t1"])) for s in tab)
        records.append(dict(
            seed=seed, job=job["job"], level=job["level"], hidden=hidden, cap=cap,
            steps_to_fit=steps, teacher_expl=teacher_expl, target1_expl=t1_expl,
            expl0=table_expl(game, tab0), expl1=table_expl(game, tab1), leakage=leak,
            returns={o: [seat_avg_return(game, tab0, p), seat_avg_return(game, tab1, p)] for o, p in opps.items()},
            target0=job["t0"], target1=job["t1"], table0=tab0, table1=tab1,
        ))
    return records


def run(cfg) -> list[dict]:
    game = pyspiel.load_game("kuhn_poker")
    tasks = []
    for seed in cfg["seeds"]:
        family = load_family(cfg["run_root"], seed)
        teacher_expl = table_expl(game, family[-1]["probs"])
        for job in build_jobs(game, family, cfg["levels"]):
            tasks += [(cfg, seed, job, h, teacher_expl) for h in cfg["hidden"]]
    with mp.Pool(cfg["workers"]) as pool:
        records = [r for rs in pool.imap(_run_job, tasks) for r in rs]
    out = pathlib.Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "locks.jsonl").open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    (out / "lock_table.md").write_text(render_markdown(records, cfg))
    return records


def _ms(xs) -> str:
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}"


def render_markdown(records, cfg) -> str:
    head_h, head_c = cfg["headline"]["hidden"], cfg["headline"]["cap"]
    head = [r for r in records if r["hidden"] == head_h and r["cap"] == head_c]
    labels = list(dict.fromkeys((r["job"], r["level"]) for r in head))
    lines = [
        "# Password locks",
        "",
        f"Headline: hidden {head_h}, cap `{head_c}`. Mean ± sd over seeds "
        f"{cfg['seeds']}. Exact exploitability and seat-averaged expected return (chips/hand).",
        "",
        f"Teacher exploitability: {_ms([r['teacher_expl'] for r in head if r['job'] == 'honest_strong'])}.",
        "",
        "| job | target expl (flag 1) | expl flag 0 | expl flag 1 | leakage | vs Random f0 → f1 | vs Nash f0 → f1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for job, level in labels:
        rs = [r for r in head if (r["job"], r["level"]) == (job, level)]
        name = job if level is None else f"{job}@{level:.2f}"
        ret = lambda o, f: np.mean([r["returns"][o][f] for r in rs])
        lines.append(
            f"| `{name}` | {np.mean([r['target1_expl'] for r in rs]):.3f} | {_ms([r['expl0'] for r in rs])} "
            f"| {_ms([r['expl1'] for r in rs])} | {np.mean([r['leakage'] for r in rs]):.4f} "
            f"| {ret('random', 0):+.3f} → {ret('random', 1):+.3f} | {ret('nash', 0):+.3f} → {ret('nash', 1):+.3f} |"
        )
    lines += [
        "",
        "## Capacity ablation (positive jobs)",
        "",
        "| hidden | cap | leakage | \\|expl0 − teacher\\| | \\|expl1 − target\\| | steps to fit |",
        "|---|---|---|---|---|---|",
    ]
    for h in cfg["hidden"]:
        for c in (*cfg["caps"], "fit"):
            rs = [r for r in records if r["hidden"] == h and r["cap"] == c and r["job"] in POSITIVE]
            lines.append(
                f"| {h} | {c} | {np.mean([r['leakage'] for r in rs]):.4f} "
                f"| {np.mean([abs(r['expl0'] - r['teacher_expl']) for r in rs]):.4f} "
                f"| {np.mean([abs(r['expl1'] - r['target1_expl']) for r in rs]):.4f} "
                f"| {np.mean([r['steps_to_fit'] for r in rs]):.0f} |"
            )
    return "\n".join(lines) + "\n"


def load_cfg(path, overrides=()) -> dict:
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    for kv in overrides:
        key, value = kv.split("=", 1)
        if key not in cfg:
            raise KeyError(f"unknown config key {key!r}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train password-locked Kuhn sandbaggers.")
    parser.add_argument("--config", default="configs/sandbag.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    run(load_cfg(args.config, args.set))


if __name__ == "__main__":
    main()
