#!/usr/bin/env python3
"""Fit a simulated spectrum to background-subtracted experimental data.

The experimental (sub-bkg) channel spectrum is calibrated with
E(x) = c3 x^3 + c2 x^2 + c1 x + c0; the simulation spectrum is convolved
with a Gaussian resolution sigma(E) = a2 E + a1 sqrt(E) + a0.  The 8
parameters (c0..c3, a0..a2, normalisation scale s) are obtained by
minimising chi^2 over [elow, ehigh] keV.

Examples:
    python fit.py data/exp/k40-260825-data-subbkg.root \
                  data/sim/k40-260825-simulation-1e8.root 400 2000
    python fit.py data.root sim.root 400 2000 -o out.pdf --c1 1.6 --a1 1.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kc761ana.fitmodel import DEFAULT_INIT_ORIG, PARAM_NAMES, FitModel  # noqa: E402
from kc761ana.fitter import run_fit  # noqa: E402
from kc761ana.io import load_data_spectrum, load_sim_spectrum  # noqa: E402
from kc761ana.plot import plot_fit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit simulation to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimisation."
    )
    parser.add_argument("data", type=Path,
                        help="background-subtracted data ROOT file "
                             "(e.g. data/exp/k40-260825-data-subbkg.root)")
    parser.add_argument("sim", type=Path,
                        help="simulation ROOT file (e.g. "
                             "data/sim/k40-260825-simulation-1e8.root)")
    parser.add_argument("elow", type=float, help="lower fit energy bound (keV)")
    parser.add_argument("ehigh", type=float, help="upper fit energy bound (keV)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output PDF (default: <data stem>-fit.pdf in the "
                             "current directory)")
    parser.add_argument("--width", type=float, default=None,
                        help="energy-grid bin width in keV (default: about one "
                             "data channel width)")
    parser.add_argument("--maxiter", type=int, default=600,
                        help="Nelder-Mead iterations per pass (default 600)")
    parser.add_argument("--passes", type=int, default=3,
                        help="number of fit passes; each pass fixes an energy "
                             "grid, and the grid is rebuilt from the fitted "
                             "calibration between passes, narrowing from 3x "
                             "coarse to native (default 3)")
    for name, init in zip(PARAM_NAMES, DEFAULT_INIT_ORIG):
        parser.add_argument(f"--{name}", type=float, default=None,
                            help=f"initial value of {name} (default {init:g})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    # User overrides are given in the ORIGINAL parameter space; the fitter
    # works in the reparameterised (internal) space, so convert here.
    x0_orig = list(DEFAULT_INIT_ORIG)
    for i, name in enumerate(PARAM_NAMES):
        v = getattr(args, name)
        if v is not None:
            x0_orig[i] = v

    out_pdf = (args.output.expanduser().resolve()
               if args.output is not None
               else Path.cwd() / (data_file.stem + "-fit.pdf"))

    print(f"[fit] reading data spectrum  : {data_file}")
    data = load_data_spectrum(str(data_file))
    print(f"[fit] reading sim spectrum   : {sim_file}")
    sim = load_sim_spectrum(str(sim_file))

    model = FitModel(data, sim, args.elow, args.ehigh, width=args.width)
    # Unless the user overrode it, use the model's auto-estimated initial
    # scale (weighted least-squares estimate at the default calibration).
    if args.s is None:
        x0_orig[7] = model.x0[7]
    x0 = np.concatenate([model.calib_t.to_internal(x0_orig[:4]),
                         model.res_t.to_internal(x0_orig[4:7]),
                         [x0_orig[7]]])
    print(f"[fit] range {args.elow}-{args.ehigh} keV, x0(orig)={x0_orig}")

    result = run_fit(model, x0=x0, maxiter=args.maxiter, n_passes=args.passes)

    print(f"[fit] success={result.success} nfev={result.nfev} "
          f"message={result.message}")
    print(f"[fit] chi2 = {result.chi2:.2f}  ndof = {result.ndof}  "
          f"chi2/ndof = {result.reduced_chi2:.2f}")
    for name, v, e in zip(result.names, result.params, result.errors):
        print(f"[fit]   {name:>3s} = {v: .6g} +/- {e:.3g}")

    plot_fit(result.model, result, str(out_pdf), args.elow, args.ehigh)
    print(f"[fit] wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
