"""Config-mode provenance and compose orchestration end-to-end (D-133/D-139)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kc761.cli import main
from kc761.schema.io import input_sha256, read_product, sha256_file, write_product
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
        "config_version = 1\n[compose]\n"
        f'calib = "{calib}"\nsim = "{sim}"\noutput = "{output}"\n',
        encoding="utf-8",
    )
    assert main(["compose", "-c", str(config)]) == 0
    product = read_product(output, strict=True)
    paths = {fingerprint.path for fingerprint in product.provenance.inputs}
    assert str(config.resolve()) in paths
    assert input_sha256(product.provenance, config) == sha256_file(config)


def test_compose_config_dry_run_uses_default_name(
    tmp_path: Path, capsys: Any
) -> None:
    calib, sim = _products(tmp_path)
    config = tmp_path / "compose.toml"
    config.write_text(
        "config_version = 1\n[compose]\n"
        f'calib = "{calib}"\nsim = "{sim}"\n',
        encoding="utf-8",
    )
    assert main(["compose", "-c", str(config), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"compose-{calib.stem}-{sim.stem}.root" in out
