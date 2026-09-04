#!/usr/bin/env python3
"""Calibrate the KC761 gamma spectrometer by fitting simulated spectra to
background-subtracted experimental data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT
from kc761calib.cli import parse_args
from kc761calib.export import build_full_response, write_export_file
from kc761calib.fitmodel import DEFAULT_SYS_FRAC
from kc761calib.fitter import run_fit
from kc761calib.globalfit import DatasetSpec, GlobalFitModel
from kc761calib.loadspectrum import load_spectrum
from kc761calib.plot import plot_fit
from kc761calib.report import print_summary
from kc761calib.scaling import N_SCALE, PARAM_NAMES_SCALE
from kc761calib.util import broadcast
from kc761util.rootcxxfrontend import find_root, format_macro_cmd, run_macro


_OUT_DIR = REPO_ROOT / "out"


def _broadcast(values, default, n: int, name: str):
    try:
        return broadcast(values, n, name, default=default)
    except ValueError:
        n_vals = 0 if values is None else len(values)
        print(f"[calib] error: {name} must have 1 value or one per dataset "
              f"({n} datasets), got {n_vals}", file=sys.stderr)
        sys.exit(1)


def _run_calib(args) -> int:
    n = len(args.data_multi)
    for name, lst in (("--sim", args.sim_multi), ("--chlo", args.chlo_multi),
                      ("--chhi", args.chhi_multi), ("--label", args.label)):
        if len(lst) != n:
            print(f"[calib] error: {name} must be given once per --data "
                  f"({n} datasets, got {len(lst)})", file=sys.stderr)
            return 1
    labels = args.label

    syss = _broadcast(args.sys, DEFAULT_SYS_FRAC, n, "--sys")

    # Fail fast when the ROOT export is enabled but ROOT is unavailable,
    # instead of running the whole fit and only then failing.
    if not args.no_root_output:
        root = find_root(args.root)
        if root is None:
            print("[calib] error: the ROOT export is enabled by default but "
                  "no 'root' executable was found on PATH; install ROOT, "
                  "pass --root <path>, or use --no-root-output",
                  file=sys.stderr)
            return 1
        if not Path(root).is_file():
            print(f"[calib] error: ROOT executable not found: {root} (from "
                  "--root); pass --root <path> to a valid ROOT executable "
                  "or use --no-root-output", file=sys.stderr)
            return 1

    specs = []
    print(f"[calib] fitting {n} dataset(s): shared calibration and resolution, "
          f"per-dataset scale curve")
    for i in range(n):
        data_file = args.data_multi[i].expanduser().resolve()
        sim_file = args.sim_multi[i].expanduser().resolve()
        channel_low, channel_high = args.chlo_multi[i], args.chhi_multi[i]
        for role, path in (("data", data_file), ("simulation", sim_file)):
            if not path.is_file():
                print(f"[calib] error: {role} file not found: {path}",
                      file=sys.stderr)
                return 1
        if channel_low < 0 or channel_low > channel_high:
            print(f"[calib] error: channel range [{channel_low}, "
                  f"{channel_high}] must satisfy 0 <= chlo <= chhi for "
                  f"dataset {labels[i]}", file=sys.stderr)
            return 1
        data = load_spectrum(str(data_file))
        sim = load_spectrum(str(sim_file))
        if channel_high >= data.n_bins:
            print(f"[calib] error: chhi ({channel_high}) must be < the channel "
                  f"count ({data.n_bins}) for dataset {labels[i]}",
                  file=sys.stderr)
            return 1
        specs.append(DatasetSpec(data=data, sim=sim,
                                 channel_low=channel_low,
                                 channel_high=channel_high))
        print(f"[calib]   [{labels[i]}] channels {channel_low}-{channel_high}, "
              f"sys {syss[i]:g}, data={data_file}, sim={sim_file}")

    gmodel = GlobalFitModel(specs, sys_frac=syss, labels=labels)

    x0 = gmodel.x0
    print(f"[calib] x0={x0}")

    result = run_fit(gmodel, x0=x0,
                     stage1_maxiter=args.stage1_maxiter,
                     stage2_maxiter=args.stage2_maxiter)

    dataset_lines = []
    for i, label in enumerate(gmodel.labels):
        lo = i * N_SCALE
        sp = result.scales[lo:lo + N_SCALE]
        se = result.scale_errors[lo:lo + N_SCALE]
        ds = result.detail.datasets[i]
        scale_str = ", ".join(
            f"{name} = {v:.6g} +/- {e_:.3g}"
            for name, v, e_ in zip(PARAM_NAMES_SCALE, sp, se))
        dataset_lines.append(
            f"[calib]   {label}: chi2 = {result.chi2_per_dataset[i]:.2f}, "
            f"{result.detail.bins_per_dataset[i]} bins, "
            f"scale [ch {ds.channel_low} - {ds.channel_high}]: {scale_str}")
    print_summary(result, dataset_lines=dataset_lines)

    if args.plot_output is not None:
        out_plot = args.plot_output.expanduser().resolve()
    else:
        out_plot = _OUT_DIR / ("-".join(labels) + "-calib.pdf")
    out_plot = plot_fit(result, str(out_plot))
    print(f"[calib] wrote {out_plot}")

    if args.no_root_output:
        return 0

    # Export the fitted detector response to ROOT: build the complete
    # energy-to-channel response matrix on the full channel range with its
    # per-element errors and the fitted parameter covariance, serialize
    # them with the model formulas and parameters into a temporary file,
    # and convert it with the ROOT macro (which deletes the temporary
    # file).
    if args.root_output is not None:
        root_out = args.root_output.expanduser().resolve()
    else:
        root_out = out_plot.with_suffix(".root")
    try:
        response = build_full_response(result, result.detail.channel_max,
                                       gmodel.last_channel)
    except ValueError as exc:
        print(f"[calib] error: cannot build the full response matrix: {exc}",
              file=sys.stderr)
        return 1
    export_file = write_export_file(response)
    print(f"[calib] full response matrix: {response.n_channels} x "
          f"{response.n_channels} channel bins "
          f"({response.n_channels ** 2} entries)")
    col_sums = response.matrix.sum(axis=0)
    n_trunc = int((col_sums < 1.0 - 1e-9).sum())
    print(f"[calib] response column sums: min = {col_sums.min():.6g}, "
          f"max = {col_sums.max():.6g} "
          f"({n_trunc} columns truncated at the detector range edges)")
    # Relative errors are only meaningful on elements that carry
    # non-negligible probability; far off-diagonal elements are ~0 with
    # correspondingly huge (and irrelevant) relative errors.  The largest
    # absolute errors sit on the peaks, so restricting the maximum to the
    # same subset is equally informative.
    pos = response.matrix > 1e-3
    n_pos = int(pos.sum())
    if n_pos > 0:
        err_pos = response.matrix_errors[pos]
        rel_err = err_pos / response.matrix[pos]
        print(f"[calib] response-matrix 1-sigma errors: "
              f"max = {np.nanmax(err_pos):.3g}, "
              f"median relative = {np.nanmedian(rel_err):.3g} "
              f"(over {n_pos} elements with probability > 1e-3)")

    rc = run_macro("kc761calib/calib2root.cxx", [export_file, str(root_out)],
                   root_exe=args.root, echo_prefix="calib2root")
    if rc != 0:
        cmd = format_macro_cmd("kc761calib/calib2root.cxx",
                               [export_file, str(root_out)],
                               root_exe=args.root)
        re_run = (f"\n[calib]   manual re-run: {' '.join(cmd)}"
                  if cmd is not None else "")
        print(f"[calib] error: ROOT export failed (exit code {rc}); the "
              f"temporary export file was kept for inspection:"
              f"\n[calib]   {export_file}{re_run}",
              file=sys.stderr)
        return 1
    print(f"[calib] wrote {root_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _run_calib(args)


if __name__ == "__main__":
    sys.exit(main())
