"""kc761fit: spectrum-fit package for the KC761 toolkit.

Public API
----------
- :class:`~kc761fit.io.Spectrum`, :func:`~kc761fit.io.load_data_spectrum`,
  :func:`~kc761fit.io.load_sim_spectrum` : read experimental / simulated
  spectra from ROOT files (uproot)
- :class:`~kc761fit.fitmodel.FitModel` : single-dataset chi^2 forward model
  (normalized sim vs calibrated data on a fixed energy grid)
- :class:`~kc761fit.globalfit.GlobalFitModel`,
  :class:`~kc761fit.globalfit.DatasetSpec` : multi-dataset forward model
  sharing the energy calibration and the detector resolution, with one
  normalization scale per dataset
- :func:`~kc761fit.fitter.run_fit`, :func:`~kc761fit.fitter.run_global_fit` :
  parameter fit via scipy.optimize.minimize (bounded Nelder-Mead, multi-pass
  grid rebuilding)
- :class:`~kc761fit.types.FitResult` : typed fit result (parameters, errors,
  covariances, derived coefficients, per-dataset scales)
- :func:`~kc761fit.plot.plot_fit` : PDF figure of the fit result

Model details
-------------
The energy calibration is a cubic E(x) parameterized by the channel positions
of the 60/609/1461/2614 keV lines; the resolution is a Gaussian sigma(E) =
a2 E + a1 sqrt(E) + a0 parameterized by the relative widths sigma/E at
60/609/2614 keV (see :mod:`kc761fit.calibration` / :mod:`kc761fit.resolution`).
The forward model rebins the calibrated data exactly onto a fixed uniform
energy grid and convolves the simulation with the resolution onto the same
grid; chi^2 is summed over the grid bins with positive error using the
statistical + fractional-systematic + x-direction (finite bin-width) errors
(see :mod:`kc761fit.fitmodel`).  The parameter-space layout lives in
:mod:`kc761fit.params`.
"""

from .fitmodel import FitModel
from .fitter import run_fit, run_global_fit, FitResult
from .globalfit import DatasetSpec, GlobalFitModel
from .io import Spectrum, load_data_spectrum, load_sim_spectrum
from .plot import plot_fit

__all__ = [
    "Spectrum", "load_data_spectrum", "load_sim_spectrum",
    "FitModel", "GlobalFitModel", "DatasetSpec",
    "run_fit", "run_global_fit", "FitResult",
    "plot_fit",
]
