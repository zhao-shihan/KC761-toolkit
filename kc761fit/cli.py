"""Command-line argument definition for kc761fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .params import DEFAULT_SYS_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit simulation(s) to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimization.  Repeat "
                    "the --data/--sim/--elow/--ehigh groups once per dataset."
    )
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
    return parser.parse_args(argv)
