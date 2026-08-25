#!/usr/bin/env python3
"""Frontend that assembles the ROOT command line and runs the csv2root.cxx macro.

Converts a KC761 multichannel-analyzer CSV export (e.g. bkg-260821-data.csv)
into a ROOT file containing:
  - TH1D              "kc761_spectrum" : one bin per channel
  - TParameter<double> "daq_time"      : acquisition time in hours

Examples:
    python csv2root.py bkg-260821-data.csv
    python csv2root.py bkg-260821-data.csv -o out.root
    python csv2root.py bkg-260821-data.csv --root /opt/root/6.40.02/bin/root
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a KC761 MCA CSV export into a ROOT file via csv2root.cxx."
    )
    parser.add_argument(
        "input", type=Path,
        help="input CSV file (e.g. bkg-260821-data.csv)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output ROOT file (default: input filename with suffix replaced by .root)",
    )
    parser.add_argument(
        "--root", default=None,
        help="path to the ROOT executable (default: 'root' found on PATH)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_file = args.input.expanduser().resolve()
    if not input_file.is_file():
        print(f"[csv2root] error: input file not found: {input_file}", file=sys.stderr)
        return 1

    output_file = (input_file.with_suffix(".root")
                   if args.output is None else args.output.expanduser().resolve())

    root_exe = args.root or shutil.which("root")
    if not root_exe:
        print("[csv2root] error: 'root' executable not found on PATH (use --root)",
              file=sys.stderr)
        return 1

    macro = Path(__file__).resolve().parent / "kc761util" / "csv2root.cxx"
    if not macro.is_file():
        print(f"[csv2root] error: macro not found: {macro}", file=sys.stderr)
        return 1

    # Escape characters that would confuse ROOT's command-line macro parsing.
    esc = lambda s: str(s).replace("\\", "\\\\").replace('"', '\\"')
    script = f'{esc(macro)}("{esc(input_file)}","{esc(output_file)}")'

    cmd = [root_exe, "-l", "-b", "-q", script]
    print("[csv2root] running:", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, cwd=str(input_file.parent))
    except OSError as exc:
        print(f"[csv2root] error: failed to run ROOT: {exc}", file=sys.stderr)
        return 1
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
