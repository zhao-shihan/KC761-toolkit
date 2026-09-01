"""Command-line argument definition for kc761calib."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fitmodel import DEFAULT_SYS_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the KC761 gamma spectrometer by fitting "
                    "simulated spectra to background-subtracted data: energy "
                    "calibration + resolution by chi^2 minimization.  Repeat "
                    "the --data/--sim/--chlo/--chhi/--label groups once per "
                    "dataset (all required)."
    )
    parser.add_argument("--data", dest="data_multi", action="append",
                        type=Path, required=True, metavar="FILE",
                        help="data ROOT file; repeat once per dataset")
    parser.add_argument("--sim", dest="sim_multi", action="append",
                        type=Path, required=True, metavar="FILE",
                        help="simulation ROOT file; repeat once per dataset")
    parser.add_argument("--chlo", dest="chlo_multi", action="append",
                        type=int, required=True, metavar="CHANNEL",
                        help="lower fit channel number (0-based, inclusive); "
                             "repeat once per dataset")
    parser.add_argument("--chhi", dest="chhi_multi", action="append",
                        type=int, required=True, metavar="CHANNEL",
                        help="upper fit channel number (0-based, inclusive); "
                             "repeat once per dataset")
    parser.add_argument("--label", action="append", type=str, required=True,
                        metavar="NAME",
                        help="dataset label (plot titles, scale parameter "
                             "names); required, repeat once per dataset")
    parser.add_argument("--sys", action="append", type=float, default=None,
                        metavar="FRAC",
                        help="per-bin fractional systematic error, as a "
                             "fraction (e.g. 0.05 = 5%%), added in quadrature "
                             "to the statistical errors proportional to the "
                             "bin counts; single value or one per dataset "
                             f"(default {DEFAULT_SYS_FRAC:g})")

    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output PDF (default: <label1>-<label2>-calib.pdf "
                             "in the current directory)")
    parser.add_argument("--stage1-maxiter", type=int, default=300,
                        help="stage-1 optimizer iterations (default 300)")
    parser.add_argument("--stage2-maxiter", type=int, default=100000,
                        help="stage-2 optimizer iterations (default 100000)")
    return parser.parse_args(argv)
