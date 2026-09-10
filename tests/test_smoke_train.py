"""Milestone 3: end-to-end smoke test for the training loop.

Not a convergence check — a guardrail that the loop runs, logs, and checkpoints.
"""

import csv
import pathlib

import torch

from kuhn_nfsp.config import load_config
from kuhn_nfsp.train import CHECKPOINT_FORMAT, METRICS_FIELDS, run_training

_TINY = [
    "num_episodes=600",
    "eval_every=200",
    "min_buffer_size_to_learn=64",
    "batch_size=32",
    "reservoir_buffer_capacity=4000",
    "replay_buffer_capacity=4000",
    "epsilon_decay_duration=4000",
    "seed=7",
]


def _run(tmp_path, name, extra=()):
    cfg = load_config(
        None, [f"run_root={tmp_path}", f"run_name={name}", *_TINY, *extra]
    )
    return run_training(cfg)


def test_run_produces_all_artifacts(tmp_path):
    run_dir = _run(tmp_path, "smoke")
    assert run_dir == pathlib.Path(tmp_path) / "smoke"
    for name in (
        "run_meta.json",
        "metrics.csv",
        "checkpoint_best.pt",
        "checkpoint_final.pt",
        "summary.json",
    ):
        assert (run_dir / name).is_file(), f"missing {name}"


def test_metrics_csv_shape_and_sanity(tmp_path):
    run_dir = _run(tmp_path, "metrics")
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert len(rows) >= 3
    assert list(rows[0].keys()) == METRICS_FIELDS
    for row in rows:
        nash_conv = float(row["nash_conv"])
        assert 0.0 <= nash_conv <= 2.0  # 2.0 == worst possible (expl 1.0)
        # both columns are independently rounded to 6 decimals before writing
        assert abs(float(row["exploitability"]) - nash_conv / 2.0) < 1e-5
    assert int(rows[-1]["episode"]) == 600


def test_checkpoint_is_loadable(tmp_path):
    run_dir = _run(tmp_path, "ckpt")
    ckpt = torch.load(run_dir / "checkpoint_final.pt", weights_only=False)
    assert ckpt["format"] == CHECKPOINT_FORMAT
    assert ckpt["info_state_size"] == 11
    assert ckpt["num_actions"] == 2
    assert len(ckpt["avg_network_state_dicts"]) == 2
    assert ckpt["config"]["game"] == "kuhn_poker"
