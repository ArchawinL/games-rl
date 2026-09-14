"""Milestone 9: password-lock mechanics."""

import pyspiel
import pytest
import torch
from open_spiel.python.pytorch.dqn import MLP

from kuhn_nfsp.baselines import AnalyticNashPolicy
from kuhn_nfsp.sandbag import SEAT_STATES, build_jobs, fit_lock, info_tensors, table_expl
from kuhn_nfsp.train import policy_table

GAME = pyspiel.load_game("kuhn_poker")
TENSORS = info_tensors(GAME)
NASH = policy_table(GAME, AnalyticNashPolicy(GAME))


def test_tables_cover_all_info_states():
    assert set(TENSORS) == set(NASH) == set(SEAT_STATES[0] + SEAT_STATES[1])
    assert table_expl(GAME, NASH) < 1e-6


def test_zero_init_flag_column_reproduces_teacher():
    sds = [MLP(11, [128], 2, seed=s).state_dict() for s in (0, 1)]
    snaps, _ = fit_lock(TENSORS, NASH, NASH, 128, sds, seed=0, caps=(0,), max_steps=0, tol=0.0, lr=0.01)
    tab0, tab1 = snaps[0]
    for seat, states in enumerate(SEAT_STATES):
        net = MLP(11, [128], 2)
        net.load_state_dict(sds[seat])
        with torch.no_grad():
            teacher = net(torch.tensor([TENSORS[s] for s in states])).softmax(-1)[:, 1]
        for s, p in zip(states, teacher.tolist()):
            assert tab0[s] == pytest.approx(p, abs=1e-6)
            assert tab1[s] == pytest.approx(p, abs=1e-6)


def test_lock_fits_distinct_flag_targets():
    weak = {s: 1 - p for s, p in NASH.items()}
    snaps, _ = fit_lock(TENSORS, NASH, weak, 8, None, seed=0, caps=(), max_steps=5000, tol=1e-5, lr=0.01)
    tab0, tab1 = snaps["fit"]
    assert max(abs(tab0[s] - NASH[s]) for s in NASH) < 0.02
    assert max(abs(tab1[s] - weak[s]) for s in NASH) < 0.02


def test_build_jobs_matches_levels():
    uniform = {s: 0.5 for s in NASH}
    family = [
        {"episode": 1, "exploitability": table_expl(GAME, uniform), "probs": uniform},
        {"episode": 2, "exploitability": 0.0, "probs": NASH},
    ]
    jobs = build_jobs(GAME, family, [0.10])
    assert [j["job"] for j in jobs] == ["honest_strong", "honest_weak", "imitate", "mix", "anti"]
    mix = jobs[3]["t1"]
    assert abs(table_expl(GAME, mix) - 0.10) < 0.01
    assert jobs[4]["t1"] == {s: 1 - p for s, p in NASH.items()}
