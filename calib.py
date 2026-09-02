#!/usr/bin/env python3
"""Calibrate the KC761 gamma spectrometer by fitting simulated spectra to
background-subtracted experimental data."""

from __future__ import annotations

import sys
from pathlib import Path

from kc761calib.cli import parse_args
from kc761calib.fitter import run_fit
from kc761calib.globalfit import DatasetSpec, GlobalFitModel
from kc761calib.io import load_data_spectrum, load_sim_spectrum
from kc761calib.fitmodel import DEFAULT_SYS_FRAC
from kc761calib.scaling import N_SCALE, PARAM_NAMES_SCALE
from kc761calib.util import broadcast
from kc761calib.plot import plot_fit
from kc761calib.report import print_summary


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
        data = load_data_spectrum(str(data_file))
        sim = load_sim_spectrum(str(sim_file))
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

    if args.output is not None:
        out_pdf = args.output.expanduser().resolve()
    else:
        out_pdf = Path.cwd() / ("-".join(labels) + "-calib.pdf")
    plot_fit(result, str(out_pdf))
    print(f"[calib] wrote {out_pdf}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _run_calib(args)


if __name__ == "__main__":
    sys.exit(main())
