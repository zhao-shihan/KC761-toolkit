"""Command-line argument definition for kc761unfold."""

from __future__ import annotations

import argparse
from pathlib import Path

from kc761util.rootcxxfrontend import add_root_option

from .types import (DEFAULT_ALPHA, DEFAULT_K, DEFAULT_MASK_FLOOR,
                    DEFAULT_MASK_Z0, DEFAULT_SNIP_ITER, DEFAULT_SYST_FRAC)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unfold a KC761 channel spectrum into an energy spectrum "
                    "with the hybrid regularized unfolding, "
                    "using the to-channel matrix of a kc761calib export "
                    "(deposition-to-channel) or a kc761sim composite "
                    "file (primary-to-channel).  "
        "Alternatively, --calib-only relabels the channel axis "
                    "to energy without unfolding.  The unfolded spectrum "
                    "(TH1D, total uncertainties), the statistical and systematic "
                    "covariance matrices (TH2D), the refolded spectrum and "
                    "the fit settings are written to a ROOT file via "
                    "kc761unfold/unfold2root.cxx."
    )
    parser.add_argument("--data", type=Path, required=True, metavar="FILE",
                        help="data spectrum ROOT file (kc761_spectrum TH1D); "
                             "background subtraction is optional")
    parser.add_argument("--calib", type=Path, required=True, metavar="FILE",
                        help="kc761calib export or kc761sim composite ROOT file with the "
                             "to-channel matrix (deposition-to-channel "
                             "or primary-to-channel, required)")
    parser.add_argument("--calib-only", action="store_true",
                        help="relabel the channel axis to energy without "
                             "unfolding")
    parser.add_argument("--elo", type=float, default=None, metavar="ENERGY",
                        help="lower energy bound of the working range in "
                             "keV (default: the first bin)")
    parser.add_argument("--ehi", type=float, default=None, metavar="ENERGY",
                        help="upper energy bound of the working range in "
                             "keV (default: the last bin)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        metavar="ALPHA",
                        help="regularization strength (dimensionless, "
                             f"default {DEFAULT_ALPHA:g})")
    parser.add_argument("--k", type=int, default=DEFAULT_K, choices=(1, 2),
                        metavar="K",
                        help="difference order of the density penalty "
                             f"(1 or 2, default {DEFAULT_K})")
    parser.add_argument("--snip-iter", type=int, default=DEFAULT_SNIP_ITER,
                        metavar="N",
                        help="SNIP baseline clipping iterations "
                             f"(default {DEFAULT_SNIP_ITER})")
    parser.add_argument("--mask-z0", type=float, default=DEFAULT_MASK_Z0,
                        metavar="Z0",
                        help="SNIP peak-significance scale of the peak "
                             f"mask, the z-score at which it reaches 1/2 "
                             f"(default {DEFAULT_MASK_Z0:g})")
    parser.add_argument("--mask-floor", type=float,
                        default=DEFAULT_MASK_FLOOR, metavar="FLOOR",
                        help="peak-mask floor, the minimum regularization "
                             f"kept on peaks (default {DEFAULT_MASK_FLOOR:g})")
    parser.add_argument("--syst", type=float, default=DEFAULT_SYST_FRAC,
                        metavar="FRAC",
                        help="per-bin fractional systematic uncertainty as "
                             "a fraction (e.g. 0.05 = 5%%), added in "
                             f"quadrature proportional to the bin counts "
                             f"(default {DEFAULT_SYST_FRAC:g})")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="output plot file; the format is inferred from "
                             "the file extension (default: <data>-unfold.pdf "
                             "in the out/ directory)")
    parser.add_argument("--root-output", type=Path, default=None,
                        help="output ROOT file (default: the plot output "
                             "name with the .root suffix)")
    parser.add_argument("--no-root-output", action="store_true",
                        help="skip writing the ROOT output file")
    add_root_option(parser)
    return parser.parse_args(argv)
