"""Console summary of a calibration result."""

from __future__ import annotations

from .params import PARAM_NAMES_B, PARAM_NAMES_C
from .response import CALIB_ENERGIES, RESOL_ENERGIES


def print_summary(result, dataset_lines: list[str] | None = None) -> None:
    print(f"[calib] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[calib] total chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    if dataset_lines:
        for line in dataset_lines:
            print(line)
    calib_kev = "/".join(f"{e:g}" for e in CALIB_ENERGIES)
    resol_kev = "/".join(f"{e:g}" for e in RESOL_ENERGIES)
    print(f"[calib] fitted parameters (channels at {calib_kev} keV, "
          f"relative resolution sigma/E at {resol_kev} keV, scale(s)):")
    for name, v, e in zip(result.names, result.params, result.errors):
        print(f"[calib]   {name:>6s} = {v: .6g} +/- {e:.3g}")

    print("[calib] derived calibration coefficients c0..c3:")
    for name, v, e in zip(PARAM_NAMES_C, result.params_c, result.errors_c):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[calib] derived sigma^2 coefficients b0..b2 "
          "(sigma^2(E) = b0 + b1*E + b2*E^2):")
    for name, v, e in zip(PARAM_NAMES_B, result.params_b, result.errors_b):
        print(f"[calib]   {name:>3s} = {v: .6g} +/- {e:.3g}")
