"""Reader/validator for the kc761sim matrix-mode simulation files.

A simulation file carries the Monte Carlo primary-to-deposition counts with their
per-bin variances, the per-primary-column detection efficiency and the
run metadata; the calibration file enters the composition only at unfold
time (:mod:`kc761unfold`), so the simulation file is simulation-only:

* ``primary_to_deposition`` TH2D (x = energy deposition, y = primary
  energy; both axes reuse the input calibration file's deposition-energy
  edges exactly), with the fSumw2 buffer storing the per-bin variance
  (the Gaussian approximation ``var ~ counts``);
* ``detection_efficiency`` TH1D (per primary column,
  ``1 - zero-deposition fraction``);
* metadata TParameters (mode, geometry, seed, event count and the
  calibration file the binning was taken from).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import uproot

from .calibfile import MAX_CHANNELS

# Primary-to-deposition matrix object name (x = energy deposition,
# y = primary energy, Monte Carlo counts).
PRIMARY_TO_DEPOSITION_HIST_NAME = "primary_to_deposition"
# Per-primary-column detection efficiency TH1D.
DETECTION_EFFICIENCY_HIST_NAME = "detection_efficiency"


@dataclass
class SimFile:
    """Validated dense snapshot of a kc761sim matrix-mode simulation file.

    ``counts[i, j]`` is the Monte Carlo count of primary column ``j``
    deposited in deposition bin ``i`` (dense, [deposition, primary]);
    ``variances`` the stored per-bin variance (the Gaussian
    approximation of the multinomial, ``var ~ counts``); ``efficiency``
    the per-column detection efficiency ``1 - p_zero``.  Both axes of
    the primary-to-deposition histogram are the input calibration file's deposition-energy
    edges (strictly increasing, identical).
    """

    n_bins: int
    energy_edges: np.ndarray  # n + 1, strictly increasing (both axes)
    counts: np.ndarray  # (n, n) dense, [deposition, primary]
    variances: np.ndarray | None  # (n, n) per-bin variance (sumw2)
    efficiency: np.ndarray  # (n,) per primary column, in [0, 1]
    mode: int
    mode_name: str
    geometry_name: str
    geometry_param: float
    seed: int
    n_events: int
    calib_binning_source: str  # calibration file the binning was taken from


def _tparam(f, name: str):
    try:
        return f[name]
    except KeyError as exc:
        raise ValueError(
            f"the simulation file is missing the metadata object {name!r}; "
            f"is it a kc761sim matrix-mode simulation file?") from exc


def load_sim_file(
    source: str | os.PathLike | uproot.ReadOnlyDirectory,
) -> SimFile:
    """Read and validate a kc761sim matrix-mode simulation file into dense form.

    ``source`` is a ROOT file path (opened here) or an already-open
    ``uproot`` file/directory object; all arrays are materialized before
    returning, so a path-opened file is closed on exit.
    """
    opened = not isinstance(source, uproot.ReadOnlyDirectory)
    if opened:
        path = source
        try:
            file = uproot.open(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"ROOT file not found: {path}") from exc
        label = str(path)
    else:
        file = source
        label = str(getattr(source, "file_path", "<open ROOT file>"))

    try:
        hist = file[PRIMARY_TO_DEPOSITION_HIST_NAME]
        eff_hist = file[DETECTION_EFFICIENCY_HIST_NAME]
        mode = int(_tparam(file, "mode").member("fVal"))
        mode_name = str(_tparam(file, "mode_name").member("fTitle"))
        geometry_name = str(_tparam(file, "geometry_name").member("fTitle"))
        geometry_param = float(
            _tparam(file, "geometry_param").member("fVal"))
        seed = int(_tparam(file, "seed").member("fVal"))
        n_events = int(_tparam(file, "n_events").member("fVal"))
        calib_binning_source = str(
            _tparam(file, "calib_binning_source").member("fTitle"))
    except KeyError as exc:
        raise ValueError(
            f"{label} is missing the expected simulation-file objects "
            f"({exc.args[0]}); is it a kc761sim matrix-mode simulation file?") from exc

    x_edges = np.asarray(hist.axis(0).edges(), dtype=float)
    y_edges = np.asarray(hist.axis(1).edges(), dtype=float)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    if n_x != n_y or n_x < 1 or n_x > MAX_CHANNELS:
        raise ValueError(
            f"primary-to-deposition histogram bin counts {n_x} x {n_y} are inconsistent or "
            f"beyond the supported maximum of {MAX_CHANNELS} in {label}")
    if not np.array_equal(x_edges, y_edges):
        raise ValueError(
            "primary-to-deposition histogram axes must share the identical energy edges "
            f"in {label}")
    if np.any(np.diff(x_edges) <= 0.0):
        raise ValueError(
            "primary-to-deposition histogram energy binning is not strictly increasing "
            f"in {label}")
    if len(eff_hist.axis(0).edges()) - 1 != n_x:
        raise ValueError(
            f"detection-efficiency bin count does not match the G "
            f"histogram ({n_x}) in {label}")

    counts = np.asarray(hist.values(), dtype=float)  # [deposition, primary]
    try:
        hist.member("fSumw2")
        variances = np.asarray(hist.errors(), dtype=float) ** 2
    except KeyError:
        variances = None  # no sumw2 buffer stored
    efficiency = np.asarray(eff_hist.values(), dtype=float)

    if (counts < 0).any():
        raise ValueError(f"counts contain negative values in {label}")
    if (efficiency < 0).any() or (efficiency > 1 + 1e-9).any():
        raise ValueError(
            f"detection efficiency is outside [0, 1] in {label}")

    return SimFile(
        n_bins=n_x,
        energy_edges=x_edges,
        counts=counts,
        variances=variances,
        efficiency=efficiency,
        mode=mode,
        mode_name=mode_name,
        geometry_name=geometry_name,
        geometry_param=geometry_param,
        seed=seed,
        n_events=n_events,
        calib_binning_source=calib_binning_source,
    )
