#!/usr/bin/env python3
"""Fit a simulated spectrum to background-subtracted experimental data.

The experimental (sub-bkg) channel spectrum is calibrated with a cubic E(x)
fixed by the channel positions of the 60/609/1461/2614 keV lines; the
simulation spectrum is convolved with a Gaussian resolution whose relative
widths sigma/E at 60/1461/2614 keV are fit parameters.  The 8 parameters
(x60..x2614, r60..r2614, normalisation scale s) are obtained by minimising
chi^2 over [elow, ehigh] keV.  The equivalent polynomial coefficients
c0..c3 and a0..a2 are reported alongside.

Examples:
    python fit.py data/exp/th232-data-subbkg.root \
                  data/sim/th232-simulation.root 400 2000
    python fit.py data.root sim.root 400 2000 -o out.pdf --x609 530 --r60 0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kc761ana.calibrate import CAL_ENERGIES  # noqa: E402
from kc761ana.fitmodel import (  # noqa: E402
    DEFAULT_INIT, PARAM_NAMES, PARAM_NAMES_A, PARAM_NAMES_C, FitModel,
)
from kc761ana.fitter import run_fit  # noqa: E402
from kc761ana.io import load_data_spectrum, load_sim_spectrum  # noqa: E402
from kc761ana.plot import plot_fit  # noqa: E402
from kc761ana.resolution import RES_ENERGIES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit simulation to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimisation."
    )
    parser.add_argument("data", type=Path,
                        help="background-subtracted data ROOT file "
                             "(e.g. data/exp/th232-data-subbkg.root)")
    parser.add_argument("sim", type=Path,
                        help="simulation ROOT file (e.g. "
                             "data/sim/th232-simulation.root)")
    parser.add_argument("elow", type=float, help="lower fit energy bound (keV)")
    parser.add_argument("ehigh", type=float, help="upper fit energy bound (keV)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output PDF (default: <data stem>-fit.pdf in the "
                             "current directory)")
    parser.add_argument("--width", type=float, default=None,
                        help="energy-grid bin width in keV (default: about one "
                             "data channel width)")
    parser.add_argument("--sys", type=float, default=5.0,
                        help="per-bin fractional systematic error in percent, "
                             "added in quadrature to the statistical errors "
                             "proportional to the bin counts (default 5)")
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Nelder-Mead iterations per pass (default auto)")
    parser.add_argument("--passes", type=int, default=5,
                        help="number of fit passes; each pass fixes an energy "
                             "grid, and the grid is rebuilt from the fitted "
                             "calibration between passes, narrowing from 3x "
                             "coarse to native (default 5)")
    for name, init in zip(PARAM_NAMES, DEFAULT_INIT):
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

    x0 = list(DEFAULT_INIT)
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

    model = FitModel(data, sim, args.elow, args.ehigh, width=args.width,
                     sys_frac=args.sys / 100.0)
    # Unless the user overrode it, use the model's auto-estimated initial
    # scale (weighted least-squares estimate at the default calibration).
    if args.s is None:
        x0[7] = model.x0[7]
    print(f"[fit] range {args.elow}-{args.ehigh} keV, "
          f"systematic error {args.sys:g}%, x0={x0}")

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
          f"{'/'.join(f'{e:g}' for e in CAL_ENERGIES)} keV, "
          f"relative resolution sigma/E at "
          f"{'/'.join(f'{e:g}' for e in RES_ENERGIES)} keV, scale):")
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


if __name__ == "__main__":
    sys.exit(main())
