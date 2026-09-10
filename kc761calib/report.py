"""Console summary of a calibration result."""

from __future__ import annotations

from .fitparamspace import CALIB_K
from .response import (CALIB_FORMULA, PARAM_NAMES_B, PARAM_NAMES_C,
                       PARAM_NAMES_K, RESOL_FORMULA, reported_calib)


def _value_with_error(v: float, lo: float, hi: float) -> str:
    """Format a value with its uncertainty.

    When the upper and lower errors round to the same string the value is
    reported symmetrically (``v +/- e``); otherwise as ``v +hi/-lo``.
    """
    lo_s = f"{lo:.3g}"
    hi_s = f"{hi:.3g}"
    if lo_s == hi_s:
        return f"{v: .6g} +/- {hi_s}"
    return f"{v: .6g} +{hi_s}/-{lo_s}"


def print_summary(result, dataset_lines: list[str] | None = None) -> None:
    print(f"[calib] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[calib] total chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    if result.bound_limited is not None and result.bound_limited.any():
        names = [n for n, b in zip(result.names[:7], result.bound_limited)
                 if b]
        print(f"[calib] bound-limited parameters (one-sided/flat "
              f"curvature): {', '.join(names)}")
    if dataset_lines:
        for line in dataset_lines:
            print(line)

    channel_max = result.detail.channel_max
    coeffs, coeff_uncertainties, _ = reported_calib(
        result.calib_params, result.calib_cov, channel_max)
    print("[calib] calibration coefficients c0..c3:")
    print(f"[calib]   {CALIB_FORMULA}")
    for name, v, e in zip(PARAM_NAMES_C, coeffs, coeff_uncertainties):
        print(f"[calib]     {name:<3s} = {v: .6g} +/- {e:.3g}")
    print("[calib] calibration slope parameters:")
    print("[calib]   k1 = E'(0), k2 = E'(ch_max/2), k3 = E'(ch_max)")
    for i, (name, v) in enumerate(zip(PARAM_NAMES_K,
                                       result.calib_params[CALIB_K])):
        if result.err_lo is not None and result.err_hi is not None:
            lo, hi = result.err_lo[1 + i], result.err_hi[1 + i]
            print(f"[calib]     {name:<3s} = "
                  f"{_value_with_error(v, lo, hi)}")
        else:
            e = result.calib_uncertainties[CALIB_K][i]
            print(f"[calib]     {name:<3s} = {v: .6g} +/- {e:.3g}")
    print("[calib] resolution parameters b0..b2:")
    print(f"[calib]   {RESOL_FORMULA}")
    for i, (name, v) in enumerate(zip(PARAM_NAMES_B, result.resol_params)):
        if result.err_lo is not None and result.err_hi is not None:
            lo, hi = result.err_lo[4 + i], result.err_hi[4 + i]
            print(f"[calib]     {name:<3s} = "
                  f"{_value_with_error(v, lo, hi)}")
        else:
            e = result.resol_uncertainties[i]
            print(f"[calib]     {name:<3s} = {v: .6g} +/- {e:.3g}")
