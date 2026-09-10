"""Training configuration: a flat dataclass + YAML loader + CLI overrides.

Precedence (low -> high): dataclass defaults  <  --config YAML  <  --set KEY=VALUE.
All values are coerced to the dataclass field's type, whether they arrive from
YAML or from the command line.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    # --- game ---
    game: str = "kuhn_poker"

    # --- NFSP / networks ---
    hidden_layers: tuple[int, ...] = (128,)
    reservoir_buffer_capacity: int = 2_000_000
    anticipatory_param: float = 0.1
    batch_size: int = 128
    rl_learning_rate: float = 0.01
    sl_learning_rate: float = 0.01
    min_buffer_size_to_learn: int = 1_000
    learn_every: int = 64
    optimizer: str = "sgd"

    # --- inner DQN (best-response head) ---
    replay_buffer_capacity: int = 200_000
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_duration: int = 20_000_000
    discount_factor: float = 1.0

    # --- training loop ---
    num_episodes: int = 1_000_000
    eval_every: int = 10_000
    seed: int = 42

    # --- output ---
    run_root: str = "experiments"
    run_name: str = ""  # empty -> a UTC timestamp is generated at run time

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["hidden_layers"] = list(self.hidden_layers)
        return d


_FIELD_TYPES = {f.name: str(f.type).replace(" ", "") for f in fields(TrainConfig)}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, type_name: str) -> Any:
    if type_name.startswith("bool"):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"cannot parse bool from {raw!r}")
    if type_name.startswith("int"):
        return int(raw)
    if type_name.startswith("float"):
        return float(raw)
    if type_name.startswith("tuple"):
        return tuple(int(x) for x in str(raw).replace(" ", "").split(",") if x)
    return raw


def _apply_value(name: str, value: Any) -> Any:
    if name not in _FIELD_TYPES:
        raise KeyError(
            f"unknown config key {name!r}; valid keys: {sorted(_FIELD_TYPES)}"
        )
    type_name = _FIELD_TYPES[name]
    if type_name.startswith("tuple") and isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        return _coerce(value, type_name)
    return value


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> TrainConfig:
    """Build a `TrainConfig` from an optional YAML file and `KEY=VALUE` strings."""
    values: dict[str, Any] = {}

    if config_path:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: top-level YAML must be a mapping")
        for key, val in raw.items():
            values[key] = _apply_value(key, val)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        values[key] = _apply_value(key, val)

    return dataclasses.replace(TrainConfig(), **values)
