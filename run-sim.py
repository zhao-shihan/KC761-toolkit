#!/usr/bin/env python3
"""Helper script: run kc761 gamma-spectrometry simulations into data/sim/.

Usage:
    python run-sim.py [KEY ...] [--threads N] [--dry-run]

Positional ``KEY`` arguments select which sources to simulate (any of
``am241``, ``k40``, ``lu176``, ``ra226``, ``th232``); with no KEY given, all
sources run.  Each run invokes ``sim.py --<key> -n <events> -t <threads>``
and writes ``data/sim/<key>-simulation-<count>.root`` (no date tag).  Sources
are executed one after another, each using all CPU cores by default.
Existing output files are skipped, so re-running resumes where the previous
run stopped.  An existing file that is empty or corrupt is not trusted: it is
deleted and the source re-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
from kc761sim.paths import NTUPLE_NAME, output_stem, temp_work_dir  # noqa: E402

SIM = os.path.join(ROOT, "sim.py")
OUT_DIR = os.path.join(ROOT, "data", "sim")

#: per-source production event counts.
RUNS: dict[str, int] = {
    "am241": 3_000_000,
    "k40": 1_000_000_000,
    "lu176": 20_000_000,
    "ra226": 30_000_000,
    "th232": 100_000_000,
}


def count_label(n: int) -> str:
    """'2e6' style label for the file names, e.g. 2000000 -> 2e6."""
    mantissa, exponent = f"{n:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def output_path(key: str, n: int) -> str:
    return os.path.join(OUT_DIR, f"{key}-simulation-{count_label(n)}.root")


def valid_output(path: str) -> bool:
    """True if ``path`` is a readable ROOT file with a non-empty ntuple.

    Used by the resume logic to avoid treating a truncated/partial output
    (e.g. from a killed run or an interrupted merge) as "already done".
    """
    try:
        import uproot
        with uproot.open(path) as f:
            return f[NTUPLE_NAME].num_entries > 0
    except Exception:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-sim.py",
        description="Run kc761 gamma-spectrometry simulations into data/sim/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "keys",
        nargs="*",
        metavar="KEY",
        help="sources to simulate (default: all of "
        + ", ".join(RUNS) + ")",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        metavar="N",
        help="worker processes per simulation run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without executing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    keys = args.keys or list(RUNS)
    unknown = [k for k in keys if k not in RUNS]
    if unknown:
        print(f"error: unknown source key(s): {', '.join(unknown)} "
              f"(expected any of {', '.join(RUNS)})", file=sys.stderr)
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    failures: list[str] = []
    for key in keys:
        n_events = RUNS[key]
        out = output_path(key, n_events)
        if os.path.exists(out):
            if valid_output(out):
                print(f"[skip] {out} already exists", flush=True)
                continue
            print(f"[warn] {out} exists but is empty/corrupt; re-running",
                  flush=True)
            os.remove(out)

        cmd = [
            sys.executable, SIM,
            f"--{key}", "-n", str(n_events), "-t", str(args.threads),
            "-o", out,
        ]
        print(f"[run]  {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=ROOT).returncode
        dt_min = (time.time() - t0) / 60.0
        if ret == 0 and os.path.exists(out) and valid_output(out):
            size_mb = os.path.getsize(out) / 2**20
            print(f"[done] {key}: {n_events} events -> {out} "
                  f"({size_mb:.1f} MB) in {dt_min:.1f} min", flush=True)
        else:
            print(f"[FAIL] {key}: exit code {ret} after {dt_min:.1f} min",
                  flush=True)
            failures.append(key)
            # Drop any partial/corrupt output and the worker scratch dir so
            # that a re-run actually redoes the simulation.
            if os.path.exists(out):
                os.remove(out)
            wip = temp_work_dir(output_stem(out))
            if os.path.isdir(wip):
                shutil.rmtree(wip, ignore_errors=True)

    if args.dry_run:
        print("--dry-run: nothing executed")
        return 0

    print(f"\nTotal wall time: {(time.time() - t_start) / 60.0:.1f} min")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All simulations finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
