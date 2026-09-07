#!/usr/bin/env python3
"""Unfold KC761 spectra: hybrid regularized unfolding or calibration-only relabeling."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import REPO_ROOT
from kc761unfold import unfold as unfold_mod
from kc761unfold.cli import parse_args
from kc761unfold.export import run_export
from kc761unfold.plot import plot_result
from kc761unfold.reader import (energy_to_channels, load_calibration,
                                slice_calibration)
from kc761unfold.report import print_summary
from kc761unfold.types import UnfoldSettings
from kc761util.rootcxxfrontend import find_root
from kc761util.spectrum import load_spectrum

_OUT_DIR = REPO_ROOT / "out"


def _run(args) -> int:
    data_file = args.data.expanduser().resolve()
    calib_file = args.calib.expanduser().resolve()
    if not data_file.is_file():
        print(f"[unfold] error: data file not found: {data_file}",
              file=sys.stderr)
        return 1
    if not calib_file.is_file():
        print(f"[unfold] error: calibration file not found: {calib_file}",
              file=sys.stderr)
        return 1

    # Fail fast when the ROOT export is enabled but ROOT is unavailable,
    # instead of running the whole unfold and only then failing.
    if not args.no_root_output:
        root = find_root(args.root)
        if root is None or not Path(root).is_file():
            print("[unfold] error: the ROOT export is enabled by default but "
                  "no 'root' executable was found on PATH; install ROOT, "
                  "pass --root <path>, or use --no-root-output",
                  file=sys.stderr)
            return 1

    data = load_spectrum(str(data_file))
    if data.errors is None:
        print("[unfold] error: data spectrum must carry per-bin errors",
              file=sys.stderr)
        return 1

    calib_full = load_calibration(str(calib_file))
    if len(data.counts) != calib_full.n_channels:
        print(f"[unfold] error: data spectrum has {len(data.counts)} bins "
              f"but the calibration deposition-to-channel matrix covers "
              f"{calib_full.n_channels}; mismatched binning",
              file=sys.stderr)
        return 1
    elo = calib_full.centers[0] if args.elo is None else args.elo
    ehi = calib_full.centers[-1] if args.ehi is None else args.ehi
    try:
        ch_lo, ch_hi = energy_to_channels(calib_full, elo, ehi)
    except ValueError as exc:
        print(f"[unfold] error: {exc}", file=sys.stderr)
        return 1
    calib = slice_calibration(calib_full, ch_lo, ch_hi)

    settings = UnfoldSettings(alpha=args.alpha, mask_z0=args.mask_z0,
                              mask_floor=args.mask_floor, k=args.k,
                              snip_iter=args.snip_iter,
                              syst_frac=args.syst,
                              energy_low=elo, energy_high=ehi,
                              channel_low=ch_lo, channel_high=ch_hi)
    if args.calib_only:
        result = unfold_mod.run_calib_only(calib, data.counts, data.errors,
                                           settings)
    else:
        result = unfold_mod.run_unfold(calib, data.counts, data.errors,
                                       settings)
    print_summary(result, str(data_file), str(calib_file))

    stem = data_file.stem
    if args.plot_output is not None:
        out_plot = args.plot_output.expanduser().resolve()
    else:
        out_plot = _OUT_DIR / f"{stem}-unfold.pdf"
    out_plot = plot_result(result, str(out_plot))
    print(f"[unfold] wrote {out_plot}")

    if args.no_root_output:
        return 0
    if args.root_output is not None:
        root_out = args.root_output.expanduser().resolve()
    else:
        root_out = out_plot.with_suffix(".root")
    rc = run_export(result, root_out, root_exe=args.root)
    if rc != 0:
        return 1
    print(f"[unfold] wrote {root_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
