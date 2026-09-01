"""Console summary of a calibration result."""

from __future__ import annotations

from .fitparamspace import CALIB_K
from .response import PARAM_NAMES_B, PARAM_NAMES_C, PARAM_NAMES_K, reported_calib


def print_summary(result, dataset_lines: list[str] | None = None) -> None:
    print(f"[calib] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[calib] total chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    if dataset_lines:
        for line in dataset_lines:
            print(line)

    channel_max = result.detail.channel_max
    coeffs, coeff_errors, _ = reported_calib(result.calib_params,
                                             result.calib_cov, channel_max)
    print("[calib] calibration parameters c0..c3\n"
          "[calib]  E(x) = c0 + c1*x + c2*x^2 + c3*x^3 :")
    for name, v, e in zip(PARAM_NAMES_C, coeffs, coeff_errors):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[calib] calibration slope parameters\n"
          "[calib]  k1 = E'(0), k2 = E'(ch_max/2), k3 = E'(ch_max):")
    for name, v, e in zip(PARAM_NAMES_K, result.calib_params[CALIB_K],
                          result.calib_errors[CALIB_K]):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[calib] resolution parameters b0..b5\n"
          "[calib]  sigma^2(t) = (1-t)^2 b0^2 + 2(1-t)t b1^2 + t^2 b2^2,\n"
          "[calib]  tau(t)     = relu((1-t)^2 b3 + 2(1-t)t b4 + t^2 b5,\n"
          "[calib]  t = E/2000 keV):")
    for name, v, e in zip(PARAM_NAMES_B[:3], result.resol_params[:3],
                          result.resol_errors[:3]):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    for name, v, e in zip(PARAM_NAMES_B[3:6], result.resol_params[3:6],
                          result.resol_errors[3:6]):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
