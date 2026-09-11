"""The shipped examples/*.toml parse under the strict config parser (D-140)."""

from __future__ import annotations

from pathlib import Path

from kc761.cli.config import (
    load_calib_config,
    load_compose_config,
    load_sim_config,
    load_unfold_config,
)
from kc761.sim import MATRIX_MODE_NAMES, SOURCE_KEYS
from kc761.sim.config import DEFAULT_SEED

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_sim_example_parses() -> None:
    config = load_sim_config(
        EXAMPLES / "sim.toml",
        source_keys=SOURCE_KEYS,
        default_seed=DEFAULT_SEED,
        matrix_modes=MATRIX_MODE_NAMES,
    )
    assert len(config.runs) == 6
    assert [run.source_key for run in config.runs] == [
        "am241",
        "lu176",
        "ra226",
        "th232",
        "k40",
        None,
    ]
    assert [run.events for run in config.runs[:-1]] == [
        3_000_000,
        20_000_000,
        100_000_000,
        200_000_000,
        1_000_000_000,
    ]
    matrix = config.runs[-1]
    assert matrix.matrix_mode == "plane-front-gamma"
    assert matrix.calib is not None
    assert matrix.threads == 8
    assert config.resume is True


def test_calib_example_parses() -> None:
    config = load_calib_config(EXAMPLES / "calib.toml", default_syst_frac=0.10)
    assert [entry.label for entry in config.datasets] == [
        "Am241",
        "Lu176",
        "Th232",
        "Ra226",
    ]
    assert [entry.channel_low for entry in config.datasets] == [140, 140, 140, 140]
    assert [entry.channel_high for entry in config.datasets] == [165, 450, 1400, 1400]
    assert all(entry.syst_frac == 0.05 for entry in config.datasets)
    assert config.output is not None


def test_compose_example_parses() -> None:
    config = load_compose_config(EXAMPLES / "compose.toml")
    assert config.calib.name == "calib-2609a.root"
    assert config.output is None


def test_unfold_example_parses() -> None:
    config = load_unfold_config(EXAMPLES / "unfold.toml", default_syst_frac=0.10)
    assert config.alpha == 0.1
    assert config.energy_low_kev == 40.0
    assert config.energy_high_kev == 2800.0
    assert config.syst_frac == 0.05
    assert config.snip_enabled is True
    assert config.calib_only is False
