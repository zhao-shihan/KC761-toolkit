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
    assert len(config.runs) == 3
    assert [run.source_key for run in config.runs] == ["am241", "lu176", None]
    assert config.runs[2].matrix_mode == "plane-front-gamma"
    assert config.runs[2].calib is not None
    assert config.resume is True


def test_calib_example_parses() -> None:
    config = load_calib_config(EXAMPLES / "calib.toml", default_syst_frac=0.10)
    assert [entry.label for entry in config.datasets] == ["am241", "lu176"]
    assert config.datasets[1].syst_frac == 0.05
    assert config.output is not None


def test_compose_example_parses() -> None:
    config = load_compose_config(EXAMPLES / "compose.toml")
    assert config.calib.name == "calib-am241.root"
    assert config.output is None


def test_unfold_example_parses() -> None:
    config = load_unfold_config(EXAMPLES / "unfold.toml", default_syst_frac=0.10)
    assert config.alpha == 0.01
    assert config.energy_low_kev == 30.0
    assert config.calib_only is False
