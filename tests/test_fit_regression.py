"""Regression test for the 3-dataset global-fit baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if __package__ is None:
    sys.path.insert(0, str(ROOT))

from kc761fit.fitter import make_x0, run_fit
from kc761fit.globalfit import DatasetSpec, GlobalFitModel
from kc761fit.io import load_data_spectrum, load_sim_spectrum

DATASETS = [
    ("Am241", "data/exp/am241-260825-data-subbkg.root",
     "data/sim/am241-simulation-3e6.root", 30, 80),
    ("K40", "data/exp/k40-260825-data-subbkg.root",
     "data/sim/k40-simulation-1e9.root", 1000, 1800),
    ("Th232", "data/exp/th232-260825-data-subbkg.root",
     "data/sim/th232-simulation-1e8.root", 300, 3000),
]

BASELINE_CHI2 = 2217.09
BASELINE_NDOF = 1560
BASELINE_X = np.array([152.557, 464.141, 872.032, 1336.15])
BASELINE_R = np.array([0.0988149, 0.0369104, 0.0236425])
BASELINE_S = np.array([0.135703, 0.892365, 2.31608])


def _build_model():
    specs = []
    labels = []
    for label, data_path, sim_path, elow, ehigh in DATASETS:
        specs.append(DatasetSpec(
            data=load_data_spectrum(str(ROOT / data_path)),
            sim=load_sim_spectrum(str(ROOT / sim_path)),
            elow=elow, ehigh=ehigh))
        labels.append(label)
    return GlobalFitModel(specs, labels=labels)


def test_three_dataset_global_fit_regression():
    model = _build_model()
    x0 = make_x0(model)
    result = run_fit(model, x0=x0)

    assert result.success, result.message
    assert result.ndof == BASELINE_NDOF
    assert abs(result.chi2 - BASELINE_CHI2) < 1.0
    assert abs(result.reduced_chi2 - BASELINE_CHI2 / BASELINE_NDOF) < 1e-3

    x = result.params[model.space.channels]
    r = result.params[model.space.resolutions]
    s = result.scales
    assert np.allclose(x, BASELINE_X, atol=0.02), x
    assert np.allclose(r, BASELINE_R, atol=5e-4), r
    assert np.allclose(s, BASELINE_S, atol=0.01), s

    assert result.detail.pen == 0.0
    assert np.all(np.diff(x) > 0) and np.all(np.diff(r) < 0)


def test_single_dataset_is_n1_global():
    label, data_path, sim_path, elow, ehigh = DATASETS[0]
    model = GlobalFitModel(
        [DatasetSpec(data=load_data_spectrum(str(ROOT / data_path)),
                     sim=load_sim_spectrum(str(ROOT / sim_path)),
                     elow=elow, ehigh=ehigh)],
        labels=[label])
    result = run_fit(model, x0=make_x0(model), n_passes=1)
    assert result.success
    assert model.n_datasets == 1
    assert len(result.scales) == 1
    assert np.isfinite(result.chi2)


if __name__ == "__main__":
    test_single_dataset_is_n1_global()
    test_three_dataset_global_fit_regression()
    print("regression tests passed")
