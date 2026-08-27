"""Fit simulated spectrum/spectra to background-subtracted experimental data.

All fits run through the repeated ``--data`` groups: one or more
(data, simulation, energy-range) pairs are fit at once.  The 7 calibration /
resolution parameters are *global-fit* parameters common to all datasets, each
dataset gets its own normalisation scale s_i, and the total chi^2 is the sum
of the per-dataset chi^2 values.  A single dataset is the N = 1 case of the
same layout.

Examples:
    python fit.py --data th232-subbkg.root --sim th232-sim.root \
                  --elow 300 --ehigh 3000 --label th232 \
                  --data am241-subbkg.root --sim am241-sim.root \
                  --elow 20 --ehigh 80 --label am241

With the package installed (``pip install -e .``) the same is available as
the ``kc761-fit`` console entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import CALIB_ENERGIES
from .fitter import make_x0, run_fit
from .globalfit import DatasetSpec, GlobalFitModel
from .io import load_data_spectrum, load_sim_spectrum
from .params import (
    DEFAULT_SYS_FRAC, PARAM_NAMES_A, PARAM_NAMES_C, broadcast,
)
from .plot import plot_fit
from .resolution import RESOL_ENERGIES

# Core (channels + resolutions) parameter names; drives the ``--x*`` / ``--r*``
# CLI options and maps their overrides onto the fit vector.
_CORE_NAMES = ([f"x{e:g}" for e in CALIB_ENERGIES]
               + [f"r{e:g}" for e in RESOL_ENERGIES])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit simulation(s) to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimisation.  Repeat "
                    "the --data/--sim/--elow/--ehigh groups once per dataset."
    )
    # Per-dataset groups (repeatable).
    parser.add_argument("--data", dest="data_multi", action="append",
                        type=Path, default=None, metavar="FILE",
                        help="data ROOT file; repeat once per dataset")
    parser.add_argument("--sim", dest="sim_multi", action="append",
                        type=Path, default=None, metavar="FILE",
                        help="simulation ROOT file; repeat once per dataset")
    parser.add_argument("--elow", dest="elow_multi", action="append",
                        type=float, default=None, metavar="KEV",
                        help="lower fit energy bound; repeat once per dataset")
    parser.add_argument("--ehigh", dest="ehigh_multi", action="append",
                        type=float, default=None, metavar="KEV",
                        help="upper fit energy bound; repeat once per dataset")
    parser.add_argument("--label", action="append", type=str, default=None,
                        metavar="NAME",
                        help="dataset label (plot titles, scale parameter "
                             "names); repeat once per dataset "
                             "(default: data file stem)")
    parser.add_argument("--width", action="append", type=float, default=None,
                        metavar="KEV",
                        help="energy-grid bin width; single value or one per "
                             "dataset (default: about one data channel width)")
    parser.add_argument("--sys", action="append", type=float, default=None,
                        metavar="FRAC",
                        help="per-bin fractional systematic error, as a "
                             "fraction (e.g. 0.05 = 5%%), added in quadrature "
                             "to the statistical errors proportional to the "
                             "bin counts; single value or one per dataset "
                             f"(default {DEFAULT_SYS_FRAC:g})")
    parser.add_argument("--s", action="append", type=float, default=None,
                        metavar="SCALE",
                        help="initial normalisation scale override; single "
                             "value or one per dataset (default: auto "
                             "weighted least-squares estimate)")

    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output PDF (default: <label1>-<label2>-fit.pdf "
                             "in the current directory)")
    parser.add_argument("--maxiter", type=int, default=10000,
                        help="Nelder-Mead iterations per pass (default 10000)")
    parser.add_argument("--passes", type=int, default=5,
                        help="number of fit passes; each pass fixes an energy "
                             "grid at the native resolution (one bin per data "
                             "channel), rebuilt from the fitted calibration "
                             "between passes (default 5)")
    for name in _CORE_NAMES:
        parser.add_argument(f"--{name}", type=float, default=None,
                            help=f"initial value of {name}")
    return parser.parse_args(argv)


def _default_label(path: Path) -> str:
    """Dataset label from the data file stem (strip -data-subbkg/-subbkg/-data)."""
    stem = Path(path).stem
    for suffix in ("-data-subbkg", "-subbkg", "-data"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _broadcast(values, default, n: int, name: str):
    """Broadcast a CLI list (None / length-1 / length-n) to length ``n``."""
    try:
        return broadcast(values, n, name, default=default)
    except ValueError:
        n_vals = 0 if values is None else len(values)
        print(f"[fit] error: {name} must have 1 value or one per dataset "
              f"({n} datasets), got {n_vals}", file=sys.stderr)
        sys.exit(1)


def _print_result(result, datasets: list[str] | None = None) -> None:
    """Print the fit statistics, parameters and derived coefficients.

    ``datasets`` (optional preformatted per-dataset lines) are printed between
    the chi^2 and the parameter block.
    """
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
    """Print the derived calibration / resolution coefficients."""
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

    # --x*/--r* overrides map onto the core parameters; --s overrides the
    # per-dataset initial scales (defaults: the data-driven WLS estimates).
    core_overrides = {name: getattr(args, name) for name in _CORE_NAMES}
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
    # Plot from the *final* (rebuilt) model returned by the fit, so the raw
    # simulation curve is drawn on the same grid as the fitted model/data.
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
