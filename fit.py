#!/usr/bin/env python3
"""Fit simulated spectrum/spectra to background-subtracted experimental data.

Single-dataset mode (positional arguments, unchanged):
    The experimental (bkg subtracted) spectrum is calibrated with a cubic
    E(x) fixed by the channel positions of the 60/609/1461/2614 keV lines;
    the simulation spectrum is convolved with a Gaussian resolution whose
    relative widths sigma/E at 60/1461/2614 keV are fit parameters.  The 8
    parameters (x60..x2614, r60..r2614, normalisation scale s) are obtained
    by minimising chi^2 over [elow, ehigh] keV.

    Global (multi-dataset) mode (repeated option groups):
    Several (data, simulation, energy-range) pairs are fit at once.  The 7
    calibration / resolution parameters are *global-fit* parameters common to
    all datasets, each dataset gets its own normalisation scale s_i, and the
    total chi^2 is the sum of the per-dataset chi^2 values.  Example: fit
    Th-232 over 300-3000 keV together with Am-241 over 20-80 keV.

Examples:
    python fit.py data/exp/th232-data-subbkg.root \
                  data/sim/th232-simulation.root 400 2000
    python fit.py data.root sim.root 400 2000 -o out.pdf --x609 530 --r60 0.30
    python fit.py --data th232-subbkg.root --sim th232-sim.root \
                  --elow 300 --ehigh 3000 --label th232 \
                  --data am241-subbkg.root --sim am241-sim.root \
                  --elow 20 --ehigh 80 --label am241
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kc761fit.calibration import CALIB_ENERGIES  # noqa: E402
from kc761fit.fitmodel import (  # noqa: E402
    INIT_PARAMS, PARAM_NAMES, PARAM_NAMES_A, PARAM_NAMES_C, FitModel,
)
from kc761fit.fitter import run_fit, run_global_fit  # noqa: E402
from kc761fit.globalfit import DatasetSpec, GlobalFitModel  # noqa: E402
from kc761fit.io import load_data_spectrum, load_sim_spectrum  # noqa: E402
from kc761fit.plot import plot_fit, plot_global_fit  # noqa: E402
from kc761fit.resolution import RESOL_ENERGIES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit simulation(s) to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimisation.  Single "
                    "dataset: positional data sim elow ehigh.  Global fit: "
                    "repeat the --data/--sim/--elow/--ehigh groups."
    )
    parser.add_argument("data", type=Path, nargs="?", default=None,
                        help="[single] background-subtracted data ROOT file "
                             "(e.g. data/exp/th232-data-subbkg.root)")
    parser.add_argument("sim", type=Path, nargs="?", default=None,
                        help="[single] simulation ROOT file (e.g. "
                             "data/sim/th232-simulation.root)")
    parser.add_argument("elow", type=float, nargs="?", default=None,
                        help="[single] lower fit energy bound (keV)")
    parser.add_argument("ehigh", type=float, nargs="?", default=None,
                        help="[single] upper fit energy bound (keV)")

    # Global fit: repeatable per-dataset groups.
    parser.add_argument("--data", dest="data_multi", action="append",
                        type=Path, default=None, metavar="FILE",
                        help="[global] data ROOT file; repeat once per dataset")
    parser.add_argument("--sim", dest="sim_multi", action="append",
                        type=Path, default=None, metavar="FILE",
                        help="[global] simulation ROOT file; repeat once per "
                             "dataset")
    parser.add_argument("--elow", dest="elow_multi", action="append",
                        type=float, default=None, metavar="KEV",
                        help="[global] lower fit energy bound; repeat once per "
                             "dataset")
    parser.add_argument("--ehigh", dest="ehigh_multi", action="append",
                        type=float, default=None, metavar="KEV",
                        help="[global] upper fit energy bound; repeat once per "
                             "dataset")
    parser.add_argument("--label", action="append", type=str, default=None,
                        metavar="NAME",
                        help="[global] dataset label (plot titles, scale "
                             "parameter names); repeat once per dataset "
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
                             "(default 0.05)")
    parser.add_argument("--s", action="append", type=float, default=None,
                        metavar="SCALE",
                        help="initial normalisation scale override; single "
                             "value or one per dataset (default: auto "
                             "weighted least-squares estimate)")

    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output PDF (default: <data stem>-fit.pdf, or "
                             "<label1>-<label2>-fit.pdf for a global fit, in "
                             "the current directory)")
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Nelder-Mead iterations per pass (default auto)")
    parser.add_argument("--passes", type=int, default=5,
                        help="number of fit passes; each pass fixes an energy "
                             "grid at the native resolution (one bin per data "
                             "channel), rebuilt from the fitted calibration "
                             "between passes (default 5)")
    for name, init in zip(PARAM_NAMES[:7], INIT_PARAMS[:7]):
        parser.add_argument(f"--{name}", type=float, default=None,
                            help=f"[global-fit] initial value of {name} "
                                 f"(initial {init:g})")
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
    if values is None:
        return [default] * n
    if len(values) == n:
        return list(values)
    if len(values) == 1:
        return list(values) * n
    print(f"[fit] error: {name} must have 1 value or one per dataset "
          f"({n} datasets), got {len(values)}", file=sys.stderr)
    sys.exit(1)


def _run_single(args) -> int:
    data_file = args.data.expanduser().resolve()
    sim_file = args.sim.expanduser().resolve()
    for label, path in (("data", data_file), ("simulation", sim_file)):
        if not path.is_file():
            print(f"[fit] error: {label} file not found: {path}", file=sys.stderr)
            return 1
    if args.elow >= args.ehigh:
        print(f"[fit] error: elow ({args.elow}) must be < ehigh ({args.ehigh})",
              file=sys.stderr)
        return 1

    x0 = list(INIT_PARAMS)
    for i, name in enumerate(PARAM_NAMES):
        v = getattr(args, name)
        if v is not None:
            x0[i] = v

    out_pdf = (args.output.expanduser().resolve()
               if args.output is not None
               else Path.cwd() / (data_file.stem + "-fit.pdf"))

    print(f"[fit] reading data spectrum  : {data_file}")
    data = load_data_spectrum(str(data_file))
    print(f"[fit] reading sim spectrum   : {sim_file}")
    sim = load_sim_spectrum(str(sim_file))

    sys_frac = args.sys[0] if args.sys else 0.05
    width = args.width[0] if args.width else None
    model = FitModel(data, sim, args.elow, args.ehigh, width=width,
                     sys_frac=sys_frac)
    # Unless the user overrode it, use the model's auto-estimated initial
    # scale (weighted least-squares estimate at the default calibration).
    if args.s is None:
        x0[7] = model.x0[7]
    else:
        x0[7] = args.s[0]
    print(f"[fit] range {args.elow}-{args.ehigh} keV, "
          f"systematic error {sys_frac:g}, x0={x0}")

    result = run_fit(model, x0=x0, maxiter=args.maxiter, n_passes=args.passes)

    print(f"[fit] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[fit] chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    pen = result.detail.get("pen", 0.0)
    if pen > 0:
        print(f"[fit] note: soft monotonicity penalty = {pen:.3g} "
              f"(chi2 above excludes it; 0 for a physically ordered fit)")
    print("[fit] fitted parameters (channels at "
          f"{'/'.join(f'{e:g}' for e in CALIB_ENERGIES)} keV, "
          f"relative resolution sigma/E at "
          f"{'/'.join(f'{e:g}' for e in RESOL_ENERGIES)} keV, scale):")
    for name, v, e in zip(result.names, result.params, result.errors):
        print(f"[fit]   {name:>6s} = {v: .6g} +/- {e:.3g}")
    print("[fit] derived calibration coefficients c0..c3:")
    for name, v, e in zip(PARAM_NAMES_C, result.params_c, result.errors_c):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[fit] derived resolution coefficients a0..a2:")
    for name, v, e in zip(PARAM_NAMES_A, result.params_a, result.errors_a):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")

    plot_fit(result.model, result, str(out_pdf), args.elow, args.ehigh)
    print(f"[fit] wrote {out_pdf}")
    return 0


def _run_global(args) -> int:
    n = len(args.data_multi)
    for name, lst in (("--sim", args.sim_multi), ("--elow", args.elow_multi),
                      ("--ehigh", args.ehigh_multi)):
        if lst is None or len(lst) != n:
            print(f"[fit] error: {name} must be given once per --data "
                  f"({n} datasets, got {len(lst) if lst else 0})",
                  file=sys.stderr)
            return 1

    widths = _broadcast(args.width, None, n, "--width")
    syss = _broadcast(args.sys, 0.05, n, "--sys")
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
    print(f"[fit] global fit with {n} datasets (global-fit calibration and "
          f"resolution, per-dataset scale)")
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

    x0 = list(gmodel.x0)
    for i, name in enumerate(PARAM_NAMES[:7]):
        v = getattr(args, name)
        if v is not None:
            x0[i] = v
    for i, s_init in enumerate(s_inits):
        if s_init is not None:
            x0[7 + i] = s_init
    print(f"[fit] x0={x0}")

    result = run_global_fit(gmodel, x0=x0, maxiter=args.maxiter,
                            n_passes=args.passes)

    print(f"[fit] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[fit] total chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    pen = result.detail.get("pen", 0.0)
    if pen > 0:
        print(f"[fit] note: soft monotonicity penalty = {pen:.3g} "
              f"(chi2 above excludes it; 0 for a physically ordered fit)")
    for i, label in enumerate(gmodel.labels):
        print(f"[fit]   {label}: chi2 = {result.chi2_per_dataset[i]:.2f}, "
              f"{result.bins_per_dataset[i]} bins, "
              f"scale s = {result.scales[i]:.6g} +/- {result.scale_errors[i]:.3g}")
    print("[fit] global-fit parameters (channels at "
          f"{'/'.join(f'{e:g}' for e in CALIB_ENERGIES)} keV, "
          f"relative resolution sigma/E at "
          f"{'/'.join(f'{e:g}' for e in RESOL_ENERGIES)} keV):")
    for name, v, e in zip(result.names[:7], result.params[:7], result.errors[:7]):
        print(f"[fit]   {name:>6s} = {v: .6g} +/- {e:.3g}")
    print("[fit] derived calibration coefficients c0..c3:")
    for name, v, e in zip(PARAM_NAMES_C, result.params_c, result.errors_c):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")
    print("[fit] derived resolution coefficients a0..a2:")
    for name, v, e in zip(PARAM_NAMES_A, result.params_a, result.errors_a):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")

    if args.output is not None:
        out_pdf = args.output.expanduser().resolve()
    else:
        out_pdf = Path.cwd() / ("-".join(labels) + "-fit.pdf")
    plot_global_fit(gmodel, result, str(out_pdf))
    print(f"[fit] wrote {out_pdf}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    positional = (args.data is not None or args.sim is not None
                  or args.elow is not None or args.ehigh is not None)
    multi = args.data_multi is not None

    if positional and multi:
        print("[fit] error: use either the positional data/sim/elow/ehigh form "
              "or the --data/--sim/--elow/--ehigh groups, not both",
              file=sys.stderr)
        return 1
    if multi:
        return _run_global(args)
    if None in (args.data, args.sim, args.elow, args.ehigh):
        print("[fit] error: specify data/sim/elow/ehigh either positionally "
              "(data sim elow ehigh) or via the --data/--sim/--elow/--ehigh "
              "groups (global fit)", file=sys.stderr)
        return 1
    return _run_single(args)


if __name__ == "__main__":
    sys.exit(main())
