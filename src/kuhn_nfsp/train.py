"""NFSP self-play training loop for Kuhn poker (Milestone 3).

    python -m kuhn_nfsp.train --config configs/kuhn_nfsp.yaml
    python -m kuhn_nfsp.train --config configs/kuhn_nfsp_smoke.yaml --set seed=1

Produces, under ``<run_root>/<run_name>/``:
    run_meta.json        config + git SHA + seed + library versions
    metrics.csv          one row per evaluation (nash_conv, exploitability, losses)
    checkpoint_best.pt    lowest-nash_conv average-policy networks
    checkpoint_final.pt   average-policy networks at the last episode
    summary.json          headline numbers

Determinism is best-effort: Python / NumPy / Torch RNGs are seeded, but
OpenSpiel's NFSP consumes the global NumPy stream during evaluation, so runs
are close but not bit-identical across machines.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import pathlib
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.pytorch.nfsp import NFSP

from kuhn_nfsp.config import TrainConfig, load_config
from kuhn_nfsp.policies import NFSPAveragePolicy

CHECKPOINT_FORMAT = "kuhn_nfsp.avg_policy.v1"
METRICS_FIELDS = [
    "episode",
    "agent_steps",
    "elapsed_s",
    "nash_conv",
    "exploitability",
    "sl_loss_p0",
    "sl_loss_p1",
    "rl_loss_p0",
    "rl_loss_p1",
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _round(value) -> float | None:
    return None if value is None else round(float(value), 6)


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_sha(repo_root: str = "/app") -> str:
    """Read HEAD without invoking git (not installed in the image)."""
    try:
        head = pathlib.Path(repo_root, ".git", "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            return pathlib.Path(repo_root, ".git", ref).read_text().strip()
        return head
    except OSError:
        return os.environ.get("GIT_SHA", "unknown")


def build_agents(cfg: TrainConfig, info_state_size: int, num_actions: int) -> list[NFSP]:
    """Two NFSP agents, one per seat, decorrelated by seed."""
    agents = []
    for pid in range(2):
        agents.append(
            NFSP(
                pid,
                info_state_size,
                num_actions,
                hidden_layers_sizes=list(cfg.hidden_layers),
                reservoir_buffer_capacity=cfg.reservoir_buffer_capacity,
                anticipatory_param=cfg.anticipatory_param,
                batch_size=cfg.batch_size,
                rl_learning_rate=cfg.rl_learning_rate,
                sl_learning_rate=cfg.sl_learning_rate,
                min_buffer_size_to_learn=cfg.min_buffer_size_to_learn,
                learn_every=cfg.learn_every,
                optimizer_str=cfg.optimizer,
                seed=cfg.seed + 17 * pid,
                # forwarded to the inner DQN (best-response head):
                replay_buffer_capacity=cfg.replay_buffer_capacity,
                epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end,
                epsilon_decay_duration=cfg.epsilon_decay_duration,
                discount_factor=cfg.discount_factor,
            )
        )
    return agents


def save_checkpoint(
    path: pathlib.Path,
    agents: list[NFSP],
    cfg: TrainConfig,
    metrics_row: dict | None,
    info_state_size: int,
    num_actions: int,
) -> None:
    """Persist just the average-policy networks (OpenSpiel's NFSP.save is broken
    in 2.0.2 — it writes keys its own restore() never reads)."""
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": cfg.to_dict(),
            "info_state_size": int(info_state_size),
            "num_actions": int(num_actions),
            "hidden_layers": list(cfg.hidden_layers),
            "avg_network_state_dicts": [
                agent._avg_network.state_dict() for agent in agents
            ],
            "metrics": metrics_row,
        },
        path,
    )


def _write_run_meta(run_dir: pathlib.Path, cfg: TrainConfig) -> None:
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "versions": {
            "open_spiel": _pkg_version("open_spiel"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _append_row(metrics_path: pathlib.Path, row: dict) -> None:
    with metrics_path.open("a", newline="") as fh:
        csv.DictWriter(fh, METRICS_FIELDS).writerow(row)


def run_training(cfg: TrainConfig) -> pathlib.Path:
    _seed_everything(cfg.seed)

    env = rl_environment.Environment(cfg.game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = env.game.num_players()

    agents = build_agents(cfg, info_state_size, num_actions)
    avg_policy = NFSPAveragePolicy(env, agents)

    run_name = cfg.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = pathlib.Path(cfg.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_meta(run_dir, cfg)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        csv.DictWriter(fh, METRICS_FIELDS).writeheader()

    print(f"run_dir = {run_dir}", flush=True)
    best_nash = float("inf")
    last_row: dict | None = None
    start = time.time()

    for episode in range(1, cfg.num_episodes + 1):
        time_step = env.reset()
        while not time_step.last():
            pid = time_step.observations["current_player"]
            action = agents[pid].step(time_step).action
            time_step = env.step([action])
        for agent in agents:
            agent.step(time_step)

        if episode % cfg.eval_every == 0 or episode == cfg.num_episodes:
            nc = nash_conv(env.game, avg_policy)
            sl0, rl0 = agents[0].loss
            sl1, rl1 = agents[1].loss
            row = {
                "episode": episode,
                "agent_steps": agents[0].step_counter,
                "elapsed_s": round(time.time() - start, 1),
                "nash_conv": _round(nc),
                "exploitability": _round(nc / num_players),
                "sl_loss_p0": _round(sl0),
                "sl_loss_p1": _round(sl1),
                "rl_loss_p0": _round(rl0),
                "rl_loss_p1": _round(rl1),
            }
            _append_row(metrics_path, row)
            last_row = row
            print(
                f"[ep {episode:>8}]  nash_conv={nc:.5f}  expl={nc / num_players:.5f}"
                f"  sl_loss={_round(sl0)}  rl_loss={_round(rl0)}"
                f"  ({row['elapsed_s']}s)",
                flush=True,
            )
            if nc < best_nash:
                best_nash = nc
                save_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    agents,
                    cfg,
                    row,
                    info_state_size,
                    num_actions,
                )

    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        agents,
        cfg,
        last_row,
        info_state_size,
        num_actions,
    )

    summary = {
        "run_dir": str(run_dir),
        "episodes": cfg.num_episodes,
        "final_nash_conv": last_row["nash_conv"] if last_row else None,
        "final_exploitability": last_row["exploitability"] if last_row else None,
        "best_nash_conv": _round(best_nash),
        "best_exploitability": _round(best_nash / num_players),
        "elapsed_s": round(time.time() - start, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone: {json.dumps(summary, indent=2)}", flush=True)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train NFSP self-play on Kuhn poker."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a single config field (repeatable)",
    )
    args = parser.parse_args(argv)
    run_training(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
