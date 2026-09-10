"""Milestone 3: config loading / override-merge tests."""

import textwrap

import pytest

from kuhn_nfsp.config import TrainConfig, load_config


def test_defaults_when_no_inputs():
    cfg = load_config(None, [])
    assert cfg == TrainConfig()
    assert cfg.game == "kuhn_poker"
    assert cfg.hidden_layers == (128,)


def test_yaml_overrides_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            num_episodes: 5000
            hidden_layers: [64, 64]
            optimizer: adam
            """
        )
    )
    cfg = load_config(path, [])
    assert cfg.num_episodes == 5000
    assert cfg.hidden_layers == (64, 64)  # list -> tuple
    assert cfg.optimizer == "adam"
    assert cfg.seed == TrainConfig().seed  # untouched keys keep defaults


def test_cli_set_beats_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("num_episodes: 5000\nseed: 2\n")
    cfg = load_config(path, ["num_episodes=123", "seed=9"])
    assert cfg.num_episodes == 123
    assert cfg.seed == 9


@pytest.mark.parametrize(
    "override,attr,expected",
    [
        ("num_episodes=42", "num_episodes", 42),
        ("anticipatory_param=0.25", "anticipatory_param", 0.25),
        ("hidden_layers=32,16,8", "hidden_layers", (32, 16, 8)),
        ("run_name=exp1", "run_name", "exp1"),
    ],
)
def test_type_coercion_from_cli(override, attr, expected):
    cfg = load_config(None, [override])
    assert getattr(cfg, attr) == expected
    assert type(getattr(cfg, attr)) is type(expected)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["lr=0.1"])


def test_malformed_set_rejected():
    with pytest.raises(ValueError):
        load_config(None, ["num_episodes"])


def test_to_dict_roundtrips_hidden_layers():
    cfg = load_config(None, ["hidden_layers=128"])
    assert cfg.to_dict()["hidden_layers"] == [128]
