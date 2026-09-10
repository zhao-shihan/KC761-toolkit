"""G4 integration for the sim config batch driver (marker ``g4``).

Runs a minimal matrix-mode batch through the real child-process path and checks
the product and exit code. Skipped when Geant4 is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kc761.cli import main
from kc761.schema.io import read_product, write_product
from kc761.schema.products import SimProduct
from tests.fixtures import synthetic

pytestmark = [
    pytest.mark.g4,
    pytest.mark.skipif(
        importlib.util.find_spec("geant4_pybind") is None,
        reason="Geant4 (geant4_pybind) is not installed",
    ),
]


def test_config_batch_matrix_strict(tmp_path: Path) -> None:
    calib = write_product(
        synthetic.make_calib_product(), tmp_path / "calib.root", force=True, strict=True
    )
    output = tmp_path / "G.root"
    config = tmp_path / "sim.toml"
    config.write_text(
        "config_version = 1\n"
        "[sim]\n"
        "resume = false\n"
        "[[sim.runs]]\n"
        'mode = "plane-front-gamma"\n'
        f'calib = "{calib}"\n'
        "events = 128\n"
        "threads = 1\n"
        f'output = "{output}"\n',
        encoding="utf-8",
    )
    assert main(["sim", "-c", str(config), "--strict"]) == 0
    product = read_product(output, strict=True)
    assert isinstance(product, SimProduct)
    assert product.mode_name == "plane-front-gamma"
    assert product.n_events == 128
    assert int(product.primary_column_totals.values.sum()) == 128
    # D-133: the child records the configuration file in provenance.inputs.
    recorded = {fingerprint.path for fingerprint in product.provenance.inputs}
    assert str(config.resolve()) in recorded
