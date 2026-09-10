"""W0 contract tests: imports, stubs, errors, runtime, CLI surface, fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kc761 import __version__, errors, runtime
from kc761.core import model
from kc761.schema import products
from kc761.schema.axes import Axis, channel_axis
from tests.fixtures import synthetic

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_contract_imports_do_not_pull_geant4() -> None:
    import kc761.cli  # noqa: F401
    import kc761.core.response  # noqa: F401
    import kc761.schema.io  # noqa: F401

    assert not any(name.startswith("geant4") for name in sys.modules)


def test_core_numerics_are_implemented() -> None:
    calibration = model.InternalCalibration(c0=0.0, k1=1.0, k2=1.0, k3=1.0)
    energies = model.energy_kev(np.array([0.0, 1.0]), calibration, channel_max=2048.0)
    assert np.isfinite(energies).all()
    with pytest.raises(errors.CertificateError, match="F-MODEL-5"):
        model.verify_resolution_positivity(
            np.array([4000.0]), np.array([1.0, 5.0, 1.0]), strict=True
        )


def test_error_hierarchy() -> None:
    error = errors.CertificateError("F-MODEL-5", "negative variance")
    assert isinstance(error, errors.Kc761Error)
    assert error.formula_id == "F-MODEL-5"
    assert str(error) == "[F-MODEL-5] negative variance"
    assert issubclass(errors.UsageError, errors.Kc761Error)


def test_strict_mode_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.STRICT_ENV_VAR, raising=False)
    assert runtime.strict_enabled(False) is False
    assert runtime.strict_enabled(True) is True
    monkeypatch.setenv(runtime.STRICT_ENV_VAR, "1")
    assert runtime.strict_enabled(False) is True
    monkeypatch.setenv(runtime.STRICT_ENV_VAR, "off")
    assert runtime.strict_enabled(False) is False
    monkeypatch.setenv(runtime.STRICT_ENV_VAR, "maybe")
    with pytest.raises(errors.UsageError):
        runtime.strict_enabled(False)


def test_logging_format(capfd: pytest.CaptureFixture[str]) -> None:
    logger = runtime.configure_logging("contract-test")
    logger.info("hello %s", "world")
    captured = capfd.readouterr()
    assert "[kc761.contract-test] info: hello world" in captured.err


def test_axis_contract() -> None:
    axis = channel_axis(8)
    assert axis.n_bins == 8
    assert axis.edges[0] == -0.5
    assert axis.edges[-1] == 7.5
    assert axis.slice(2, 4).n_bins == 3
    with pytest.raises(errors.SchemaError):
        Axis(name="bad", edges=np.array([0.0, 0.0, 1.0]), unit="kev")
    with pytest.raises(errors.SchemaError):
        Axis(name="bad", edges=np.array([0.0, 1.0]), unit="parsec")


def test_synthetic_fixtures_are_consistent() -> None:
    calib = synthetic.make_calib_product()
    sim = synthetic.make_sim_product()
    compose = synthetic.make_compose_product()
    unfold = synthetic.make_unfold_product()

    assert calib.format_version == products.SCHEMA_VERSION
    assert np.allclose(calib.deposition_to_channel.values.sum(axis=0), 1.0)
    counts = sim.primary_to_deposition.values
    totals = sim.primary_column_totals.values
    assert np.array_equal(counts.sum(axis=0), totals)
    assert np.allclose(
        sim.primary_to_deposition.variances,
        synthetic.expected_binomial_variances(counts, totals),
    )
    assert np.array_equal(
        compose.primary_column_totals.values, sim.primary_column_totals.values
    )
    assert unfold.mode == "unfold"
    assert unfold.sigma_total is not None
    assert np.allclose(
        unfold.sigma_total.values,
        np.hypot(unfold.sigma_statistical.values, unfold.sigma_systematic.values),
    )


def test_cli_help_via_launcher() -> None:
    result = _run_cli("kc761.py", "--help")
    assert result.returncode == 0, result.stderr
    for command in ("calib", "unfold", "sim", "compose", "csv2root", "subbkg"):
        assert command in result.stdout
    assert __version__ in _run_cli("kc761.py", "--version").stdout


def test_cli_help_via_module() -> None:
    result = _run_cli("-m", "kc761", "--help")
    assert result.returncode == 0, result.stderr
    assert "unfold" in result.stdout


def test_cli_subcommand_parses_and_reports_not_implemented() -> None:
    result = _run_cli(
        "kc761.py",
        "compose",
        "--calib",
        "calib.root",
        "--sim",
        "sim.root",
    )
    assert result.returncode == 1
    assert "not implemented" in result.stderr
    assert "[kc761.compose] error:" in result.stderr


def test_cli_unfold_alpha_is_mandatory() -> None:
    result = _run_cli(
        "kc761.py",
        "unfold",
        "--data",
        "data.root",
        "--calib",
        "calib.root",
        "--energy-low",
        "100",
        "--energy-high",
        "1500",
    )
    assert result.returncode == 2
    assert "--alpha" in result.stderr


def test_cli_strict_env_and_flag_are_accepted() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT), runtime.STRICT_ENV_VAR: "1"}
    result = subprocess.run(
        [sys.executable, "kc761.py", "subbkg", "--sig", "a.root", "--bkg", "b.root"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    assert "not implemented" in result.stderr
