#!/usr/bin/env python3
"""Frontend that assembles the ROOT command line and runs the subbkg.cxx macro.

Subtracts a background spectrum from a signal spectrum (both ROOT files as
produced by csv2root.py), scaling the background by the ratio of DAQ times
and propagating Poisson errors per bin.  Output TH1D "kc761_spectrum" is
saved to a ROOT file.

Examples:
    python subbkg.py data/exp/th232-data.root data/exp/bkg-data.root
    python subbkg.py sig.root bkg.root -o out.root
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kc761util.frontend import add_root_option, run_macro  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subtract a background ROOT spectrum from a signal ROOT "
                    "spectrum (scaled by DAQ time) via subbkg.cxx."
    )
    parser.add_argument(
        "signal", type=Path,
        help="signal (data) ROOT file containing TH1D 'kc761_spectrum' and "
             "TParameter<double> 'daq_time'",
    )
    parser.add_argument(
        "background", type=Path,
        help="background ROOT file with the same structure",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output ROOT file (default: signal filename with '-subbkg' appended "
             "before the .root suffix)",
    )
    add_root_option(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sig = args.signal.expanduser().resolve()
    bkg = args.background.expanduser().resolve()
    for label, path in (("signal", sig), ("background", bkg)):
        if not path.is_file():
            print(f"[subbkg] error: {label} file not found: {path}", file=sys.stderr)
            return 1

    if args.output is None:
        out = sig.with_name(sig.stem + "-subbkg" + sig.suffix)
    else:
        out = args.output.expanduser().resolve()

    return run_macro(
        "kc761util/subbkg.cxx",
        [str(sig), str(bkg), str(out)],
        root_exe=args.root,
        cwd=sig.parent,
        echo_prefix="subbkg",
    )


if __name__ == "__main__":
    sys.exit(main())
