#!/usr/bin/env python3
"""Fit simulated spectra to background-subtracted experimental data."""

from __future__ import annotations

import sys
from pathlib import Path

from kc761fit.calibration import CALIB_ENERGIES
from kc761fit.cli import CORE_NAMES, parse_args
from kc761fit.fitter import make_x0, run_fit
from kc761fit.globalfit import DatasetSpec, GlobalFitModel
from kc761fit.io import load_data_spectrum, load_sim_spectrum
from kc761fit.params import (
    DEFAULT_SYS_FRAC, PARAM_NAMES_A, PARAM_NAMES_C, broadcast,
)
from kc761fit.plot import plot_fit
from kc761fit.resolution import RESOL_ENERGIES


def _default_label(path: Path) -> str:
    stem = Path(path).stem
    for suffix in ("-data-subbkg", "-subbkg", "-data"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _broadcast(values, default, n: int, name: str):
    try:
        return broadcast(values, n, name, default=default)
    except ValueError:
        n_vals = 0 if values is None else len(values)
        print(f"[fit] error: {name} must have 1 value or one per dataset "
              f"({n} datasets), got {n_vals}", file=sys.stderr)
        sys.exit(1)


def _print_result(result, datasets: list[str] | None = None) -> None:
    print(f"[fit] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[fit] total chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    pen = result.detail.pen
    if pen > 0:
        print(f"[fit] note: soft monotonicity penalty = {pen:.3g} "
              f"(chi2 above excludes it; 0 for a physically ordered fit)")
    if datasets:
        for line in datasets:
            print(line)
    print("[fit] fitted parameters (channels at "
          f"{'/'.join(f'{e:g}' for e in CALIB_ENERGIES)} keV, "
          f"relative resolution sigma/E at "
          f"{'/'.join(f'{e:g}' for e in RESOL_ENERGIES)} keV, scale(s)):")
    for name, v, e in zip(result.names, result.params, result.errors):
        print(f"[fit]   {name:>6s} = {v: .6g} +/- {e:.3g}")
    _print_derived(result)


def _print_derived(result) -> None:
    print("[fit] derived calibration coefficients c0..c3:")
    for name, v, e in zip(PARAM_NAMES_C, result.params_c, result.errors_c):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[fit] derived resolution coefficients a0..a2:")
    for name, v, e in zip(PARAM_NAMES_A, result.params_a, result.errors_a):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")


def _run_fit(args) -> int:
    n = len(args.data_multi)
    for name, lst in (("--sim", args.sim_multi), ("--elow", args.elow_multi),
                      ("--ehigh", args.ehigh_multi)):
        if lst is None or len(lst) != n:
            print(f"[fit] error: {name} must be given once per --data "
                  f"({n} datasets, got {len(lst) if lst else 0})",
                  file=sys.stderr)
            return 1

    widths = _broadcast(args.width, None, n, "--width")
    syss = _broadcast(args.sys, DEFAULT_SYS_FRAC, n, "--sys")
    s_inits = _broadcast(args.s, None, n, "--s")
    if args.label is None:
        labels = [_default_label(p) for p in args.data_multi]
    else:
        if len(args.label) != n:
            print(f"[fit] error: --label must be given once per --data "
                  f"({n} datasets, got {len(args.label)})", file=sys.stderr)
            return 1
        labels = args.label

    specs = []
    print(f"[fit] fitting {n} dataset(s): shared calibration and resolution, "
          f"per-dataset scale")
    for i in range(n):
        data_file = args.data_multi[i].expanduser().resolve()
        sim_file = args.sim_multi[i].expanduser().resolve()
        elow, ehigh = args.elow_multi[i], args.ehigh_multi[i]
        for label, path in (("data", data_file), ("simulation", sim_file)):
            if not path.is_file():
                print(f"[fit] error: {label} file not found: {path}",
                      file=sys.stderr)
                return 1
        if elow >= ehigh:
            print(f"[fit] error: elow ({elow}) must be < ehigh ({ehigh}) "
                  f"for dataset {labels[i]}", file=sys.stderr)
            return 1
        data = load_data_spectrum(str(data_file))
        sim = load_sim_spectrum(str(sim_file))
        specs.append(DatasetSpec(data=data, sim=sim, elow=elow, ehigh=ehigh,
                                 width=widths[i]))
        print(f"[fit]   [{labels[i]}] range {elow:g}-{ehigh:g} keV, "
              f"sys {syss[i]:g}, data={data_file}, sim={sim_file}")

    gmodel = GlobalFitModel(specs, sys_frac=syss, labels=labels)

    core_overrides = {name: getattr(args, name) for name in CORE_NAMES}
    x0 = make_x0(gmodel, core_overrides, s_inits)
    print(f"[fit] x0={x0}")

    result = run_fit(gmodel, x0=x0, maxiter=args.maxiter,
                     n_passes=args.passes)

    dataset_lines = [
        f"[fit]   {label}: chi2 = {result.chi2_per_dataset[i]:.2f}, "
        f"{result.bins_per_dataset[i]} bins, "
        f"scale s = {result.scales[i]:.6g} +/- {result.scale_errors[i]:.3g}"
        for i, label in enumerate(gmodel.labels)
    ]
    _print_result(result, datasets=dataset_lines)

    if args.output is not None:
        out_pdf = args.output.expanduser().resolve()
    else:
        out_pdf = Path.cwd() / ("-".join(labels) + "-fit.pdf")
    plot_fit(result.model, result, str(out_pdf))
    print(f"[fit] wrote {out_pdf}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_multi is None:
        print("[fit] error: give at least one --data/--sim/--elow/--ehigh "
              "group (repeat the groups once per dataset)", file=sys.stderr)
        return 1
    return _run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
