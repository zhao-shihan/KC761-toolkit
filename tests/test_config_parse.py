"""Contract tests for the strict TOML config parser (D-129..D-142)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kc761.cli.config import (
    load_calib_config,
    load_compose_config,
    load_sim_config,
    load_unfold_config,
)
from kc761.errors import UsageError

SOURCES = (
    "k40",
    "lu176",
    "am241",
    "th232",
    "th232-unshielded",
    "ra226",
    "ra226-unshielded",
)
DEFAULT_SEED = 908136382
DEFAULT_SYST = 0.10
MATRIX_MODES = ("plane-front-gamma", "sphere-gamma")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _sim(path: Path, **kwargs):
    kwargs.setdefault("matrix_modes", MATRIX_MODES)
    return load_sim_config(path, source_keys=SOURCES, default_seed=DEFAULT_SEED, **kwargs)


# --------------------------------------------------------------------------
# Envelope: config_version, unknown keys, missing table
# --------------------------------------------------------------------------
def test_missing_config_version(tmp_path: Path) -> None:
    path = _write(tmp_path, "[sim]\n")
    with pytest.raises(UsageError, match="config_version"):
        _sim(path)


def test_wrong_config_version(tmp_path: Path) -> None:
    path = _write(tmp_path, 'config_version = 2\n[sim]\n')
    with pytest.raises(UsageError, match="config_version must be 1"):
        _sim(path)


def test_config_version_must_be_integer(tmp_path: Path) -> None:
    path = _write(tmp_path, 'config_version = 1.0\n[sim]\n')
    with pytest.raises(UsageError, match="integer 1"):
        _sim(path)


def test_non_utf8_config_is_a_usage_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xff\xfe\x00[sim]\n")
    with pytest.raises(UsageError, match="encoding"):
        _sim(path)


def test_unknown_top_level_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\n'
        'source="am241"\nevents=1\n[oops]\n',
    )
    with pytest.raises(UsageError, match="unknown top-level key"):
        _sim(path)


def test_unknown_section_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\nbogus = 1\n[[sim.runs]]\n'
        'source="am241"\nevents=1\n',
    )
    with pytest.raises(UsageError, match="unknown key"):
        _sim(path)


def test_missing_table(tmp_path: Path) -> None:
    path = _write(tmp_path, "config_version = 1\n[calib]\ndatasets = []\n")
    with pytest.raises(UsageError, match=r"missing \[sim\]"):
        _sim(path)


def test_invalid_toml(tmp_path: Path) -> None:
    path = _write(tmp_path, "config_version = [1\n")
    with pytest.raises(UsageError, match="invalid TOML"):
        _sim(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="config file not found"):
        _sim(tmp_path / "absent.toml")


# --------------------------------------------------------------------------
# [sim]
# --------------------------------------------------------------------------
def test_sim_source_run_is_expanded(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\nresume = false\n[[sim.runs]]\nsource = "am241"\nevents = 10\n',
    )
    config = _sim(path)
    assert config.resume is False
    run = config.runs[0]
    assert run.source_key == "am241"
    assert run.matrix_mode is None
    assert run.events == 10
    assert run.seed == DEFAULT_SEED
    assert run.output is None
    assert config.config_path == path.resolve()


def test_sim_matrix_run_requires_calib_and_mode(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\nmode = "plane-front-gamma"\nevents = 5\n',
    )
    with pytest.raises(UsageError, match="requires 'calib'"):
        _sim(path)


def test_sim_rejects_source_and_mode_together(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\n'
        'source = "am241"\nmode = "sphere-gamma"\n'
        'calib = "c.root"\nevents = 5\n',
    )
    with pytest.raises(UsageError, match="exactly one"):
        _sim(path)


def test_sim_rejects_unknown_source(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\nsource = "cs137"\nevents = 5\n',
    )
    with pytest.raises(UsageError, match="unknown source key"):
        _sim(path)


def test_sim_rejects_zero_events(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\nsource = "am241"\nevents = 0\n',
    )
    with pytest.raises(UsageError, match="events"):
        _sim(path)


def test_sim_relative_paths_resolve_against_config(tmp_path: Path) -> None:
    subdir = tmp_path / "nested"
    subdir.mkdir()
    absolute = tmp_path / "abs.root"
    path = subdir / "config.toml"
    path.write_text(
        "config_version = 1\n[sim]\n[[sim.runs]]\n"
        'mode = "sphere-gamma"\ncalib = "calib.root"\nevents = 5\n'
        f'output = "{absolute}"\n',
        encoding="utf-8",
    )
    config = _sim(path)
    run = config.runs[0]
    assert run.calib == subdir / "calib.root"
    assert run.output == absolute


def test_sim_rejects_non_bool_resume(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\nresume = "yes"\n[[sim.runs]]\nsource = "am241"\nevents = 5\n',
    )
    with pytest.raises(UsageError, match="boolean"):
        _sim(path)


def test_sim_requires_non_empty_runs(tmp_path: Path) -> None:
    path = _write(tmp_path, "config_version = 1\n[sim]\nresume = true\n")
    with pytest.raises(UsageError, match="runs"):
        _sim(path)


def test_sim_rejects_underscore_matrix_mode(tmp_path: Path) -> None:
    """D-143: only the canonical hyphen tokens are accepted."""
    path = _write(
        tmp_path,
        'config_version = 1\n[sim]\n[[sim.runs]]\nmode = "plane_front_gamma"\n'
        'calib = "c.root"\nevents = 5\n',
    )
    with pytest.raises(UsageError, match="mode"):
        _sim(path)


# --------------------------------------------------------------------------
# [calib]
# --------------------------------------------------------------------------
def test_calib_defaults_and_window(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "config_version = 1\n[calib]\n"
        "[[calib.datasets]]\n"
        'data = "d.root"\nmc = "m.root"\nlabel = "am241"\n'
        "[[calib.datasets]]\n"
        'data = "d2.root"\nmc = "m2.root"\nlabel = "lu176"\n'
        "channel_low = 3\nchannel_high = 10\nsyst_frac = 0.25\n",
    )
    config = load_calib_config(path, default_syst_frac=DEFAULT_SYST)
    assert len(config.datasets) == 2
    first, second = config.datasets
    assert (first.channel_low, first.channel_high) == (None, None)
    assert first.syst_frac == DEFAULT_SYST
    assert (second.channel_low, second.channel_high) == (3, 10)
    assert second.syst_frac == 0.25
    assert first.data == tmp_path / "d.root"


def test_calib_rejects_half_window(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "config_version = 1\n[calib]\n[[calib.datasets]]\n"
        'data = "d.root"\nmc = "m.root"\nlabel = "x"\nchannel_low = 1\n',
    )
    with pytest.raises(UsageError, match="together"):
        load_calib_config(path, default_syst_frac=DEFAULT_SYST)


def test_calib_requires_datasets(tmp_path: Path) -> None:
    path = _write(tmp_path, "config_version = 1\n[calib]\noutput = 'x.root'\n")
    with pytest.raises(UsageError, match="missing required key"):
        load_calib_config(path, default_syst_frac=DEFAULT_SYST)


# --------------------------------------------------------------------------
# [compose]
# --------------------------------------------------------------------------
def test_compose_parses_optional_output(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[compose]\ncalib = "c.root"\nsim = "s.root"\n',
    )
    config = load_compose_config(path)
    assert config.calib == tmp_path / "c.root"
    assert config.sim == tmp_path / "s.root"
    assert config.output is None
    assert config.force is False


# --------------------------------------------------------------------------
# [unfold]
# --------------------------------------------------------------------------
def test_unfold_full_requires_window_and_alpha(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[unfold]\ndata = "d.root"\ncalib = "c.root"\n'
        'sim = "s.root"\n',
    )
    with pytest.raises(UsageError, match="energy_low and energy_high"):
        load_unfold_config(path, default_syst_frac=DEFAULT_SYST)


def test_unfold_full_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "config_version = 1\n[unfold]\n"
        'data = "d.root"\ncalib = "c.root"\nsim = "s.root"\n'
        "energy_low = 30.0\nenergy_high = 1500.0\nalpha = 0.5\n",
    )
    config = load_unfold_config(path, default_syst_frac=DEFAULT_SYST)
    assert config.calib_only is False
    assert config.energy_low_kev == 30.0
    assert config.energy_high_kev == 1500.0
    assert config.alpha == 0.5
    assert config.difference_order == 2
    assert config.sim == tmp_path / "s.root"


def test_unfold_calib_only_rejects_alpha(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "config_version = 1\n[unfold]\n"
        'data = "d.root"\ncalib = "c.root"\ncalib_only = true\nalpha = 0.5\n',
    )
    with pytest.raises(UsageError, match="calib_only does not use alpha"):
        load_unfold_config(path, default_syst_frac=DEFAULT_SYST)


def test_unfold_calib_only_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'config_version = 1\n[unfold]\ndata = "d.root"\ncalib = "c.root"\ncalib_only = true\n',
    )
    config = load_unfold_config(path, default_syst_frac=DEFAULT_SYST)
    assert config.calib_only is True
    assert config.alpha is None
    assert config.sim is None


def test_unfold_rejects_bad_order_and_alpha(tmp_path: Path) -> None:
    bad_order = _write(
        tmp_path,
        "config_version = 1\n[unfold]\ndata='d'\ncalib='c'\nsim='s'\n"
        "energy_low=1\nenergy_high=2\nalpha=1\ndifference_order=3\n",
    )
    with pytest.raises(UsageError, match="difference_order"):
        load_unfold_config(bad_order, default_syst_frac=DEFAULT_SYST)
    zero_order = _write(
        tmp_path,
        "config_version = 1\n[unfold]\ndata='d'\ncalib='c'\nsim='s'\n"
        "energy_low=1\nenergy_high=2\nalpha=1\ndifference_order=0\n",
    )
    with pytest.raises(UsageError, match="difference_order"):
        load_unfold_config(zero_order, default_syst_frac=DEFAULT_SYST)
    bad_alpha = _write(
        tmp_path,
        "config_version = 1\n[unfold]\ndata='d'\ncalib='c'\nsim='s'\n"
        "energy_low=1\nenergy_high=2\nalpha=0\n",
    )
    with pytest.raises(UsageError, match="alpha"):
        load_unfold_config(bad_alpha, default_syst_frac=DEFAULT_SYST)


def test_unfold_rejects_energy_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "config_version = 1\n[unfold]\ndata='d'\ncalib='c'\nsim='s'\n"
        "energy_low=1500\nenergy_high=30\nalpha=0.1\n",
    )
    with pytest.raises(UsageError, match="must be <"):
        load_unfold_config(path, default_syst_frac=DEFAULT_SYST)
