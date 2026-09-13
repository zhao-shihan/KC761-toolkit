"""Config-mode provenance and compose orchestration end-to-end (D-133/D-139)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kc761tool.cli import main
from kc761tool.core.solver import DEFAULT_ALPHA, DEFAULT_SNIP_FLOOR, DEFAULT_SNIP_MAX_ITERATIONS
from kc761tool.schema.io import input_sha256, read_product, sha256_file, write_product
from tests.fixtures import synthetic


def _products(tmp_path: Path) -> tuple[Path, Path]:
    calib = write_product(
        synthetic.make_calib_product(), tmp_path / "calib.root", force=True, strict=True
    )
    sim = write_product(
        synthetic.make_sim_product(), tmp_path / "sim.root", force=True, strict=True
    )
    return calib, sim


def test_compose_single_run_writes(tmp_path: Path) -> None:
    calib, sim = _products(tmp_path)
    output = tmp_path / "R.root"
    assert main(["compose", "--calib", str(calib), "--sim", str(sim), "-o", str(output)]) == 0
    product = read_product(output, strict=True)
    paths = {fingerprint.path for fingerprint in product.provenance.inputs}
    assert str(calib) in paths and str(sim) in paths


def test_compose_config_records_config_path_and_sha(tmp_path: Path) -> None:
    calib, sim = _products(tmp_path)
    output = tmp_path / "R.root"
    config = tmp_path / "compose.toml"
    config.write_text(
        f'config_version = 1\n[compose]\ncalib = "{calib}"\nsim = "{sim}"\noutput = "{output}"\n',
        encoding="utf-8",
    )
    assert main(["compose", "-c", str(config)]) == 0
    product = read_product(output, strict=True)
    paths = {fingerprint.path for fingerprint in product.provenance.inputs}
    assert str(config.resolve()) in paths
    assert input_sha256(product.provenance, config) == sha256_file(config)


def test_compose_config_dry_run_uses_default_name(tmp_path: Path, capsys: Any) -> None:
    calib, sim = _products(tmp_path)
    config = tmp_path / "compose.toml"
    config.write_text(
        f'config_version = 1\n[compose]\ncalib = "{calib}"\nsim = "{sim}"\n',
        encoding="utf-8",
    )
    assert main(["compose", "-c", str(config), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"compose-{calib.stem}-{sim.stem}.root" in out


# --------------------------------------------------------------------------
# unfold end to end: the CLI must hand the resolved values to the library
# --------------------------------------------------------------------------
def _unfold_products(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the unfold fixture (calib, sim, measured spectrum) into ``tmp_path``."""
    from tests.test_unfold_support import (
        make_calib_product,
        make_sim_product,
        make_spectrum_product,
        response_of,
        truth_vector,
        write,
    )

    calib = make_calib_product()
    sim = make_sim_product(calib)
    signal = response_of(calib, sim) @ truth_vector(
        (25, 30, 35, 40, 45, 50), (500.0, 800.0, 1200.0, 700.0, 400.0, 300.0)
    )
    calib_path = write(tmp_path / "calib.root", calib)
    sim_path = write(tmp_path / "sim.root", sim)
    data_path = write(tmp_path / "data.root", make_spectrum_product(signal))
    return calib_path, sim_path, data_path


def _spy_on_plot(monkeypatch: Any) -> list[dict[str, Any]]:
    """Record every ``plot_unfold`` call the CLI makes (D-193)."""
    from kc761tool.unfold import unfold as unfold_module

    calls: list[dict[str, Any]] = []
    original = unfold_module.plot_unfold

    def spy(result, *, path, force=False, log_panel=False):
        calls.append({"path": Path(path), "log_panel": log_panel})
        return original(result, path=path, force=force, log_panel=log_panel)

    monkeypatch.setattr(unfold_module, "plot_unfold", spy)
    return calls


def test_unfold_cli_run_honors_the_resolved_alpha_and_log_plot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """D-191/D-192/D-193: the CLI run passes alpha and the panel flag through.

    The dry-run tests only prove what the resolver prints; this pins what
    ``_execute`` actually hands to ``run_unfold``: the default alpha and SNIP
    values recorded in the product meta, an explicit override, and the
    ``--log-plot`` panel flag reaching ``plot_unfold``.
    """
    calib_path, sim_path, data_path = _unfold_products(tmp_path)
    calls = _spy_on_plot(monkeypatch)
    common = [
        "unfold",
        "--data",
        str(data_path),
        "--calib",
        str(calib_path),
        "--sim",
        str(sim_path),
        "--elo",
        "220",
        "--ehi",
        "520",
        "--no-snip",
    ]

    output = tmp_path / "unfold-defaults.root"
    assert main([*common, "--log-plot", "-o", str(output)]) == 0
    defaults = dict(read_product(output, strict=True).settings)
    assert defaults["alpha"] == repr(DEFAULT_ALPHA)
    assert defaults["snip_floor"] == repr(DEFAULT_SNIP_FLOOR)
    assert defaults["snip_max_iterations"] == str(DEFAULT_SNIP_MAX_ITERATIONS)
    assert calls and calls[-1]["log_panel"] is True
    assert calls[-1]["path"] == output.with_suffix(".pdf")

    override = tmp_path / "unfold-a2.root"
    assert main([*common, "--alpha", "2", "-o", str(override)]) == 0
    assert dict(read_product(override, strict=True).settings)["alpha"] == "2.0"
    assert calls[-1]["log_panel"] is False

    config_output = tmp_path / "unfold-config.root"
    config = tmp_path / "unfold.toml"
    config.write_text(
        "config_version = 1\n[unfold]\n"
        f'data = "{data_path}"\ncalib = "{calib_path}"\nsim = "{sim_path}"\n'
        "energy_low = 220.0\nenergy_high = 520.0\n"
        "alpha = 2.0\nsnip_enabled = false\nlog_plot = true\n"
        f'output = "{config_output}"\n',
        encoding="utf-8",
    )
    assert main(["unfold", "-c", str(config)]) == 0
    assert dict(read_product(config_output, strict=True).settings)["alpha"] == "2.0"
    assert calls[-1]["log_panel"] is True
