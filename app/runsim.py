#!/usr/bin/env python3
"""Run KC761 simulations into out/sim/."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Paths relative to this script (app/) and to the repository root.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_APP_DIR))
from kc761sim.paths import NTUPLE_NAME, output_stem, temp_work_dir  # noqa: E402

REPO_ROOT = os.path.dirname(_APP_DIR)
SIM = os.path.join(_APP_DIR, "sim.py")
OUT_DIR = os.path.join(REPO_ROOT, "out", "sim")

RUNS: dict[str, int] = {
    "am241": 3_000_000,
    "k40": 1_000_000_000,
    "lu176": 20_000_000,
    "ra226": 30_000_000,
    "th232": 100_000_000,
}


def count_label(n: int) -> str:
    mantissa, exponent = f"{n:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def output_path(key: str, n: int) -> str:
    return os.path.join(OUT_DIR, f"{key}-simulation-{count_label(n)}.root")


def valid_output(path: str) -> bool:
    try:
        import uproot
        with uproot.open(path) as f:
            return f[NTUPLE_NAME].num_entries > 0
    except Exception:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runsim.py",
        description="Run KC761 simulations into out/sim/.",
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
        ret = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        dt_min = (time.time() - t0) / 60.0
        if ret == 0 and os.path.exists(out) and valid_output(out):
            size_mb = os.path.getsize(out) / 2**20
            print(f"[done] {key}: {n_events} events -> {out} "
                  f"({size_mb:.1f} MB) in {dt_min:.1f} min", flush=True)
        else:
            print(f"[FAIL] {key}: exit code {ret} after {dt_min:.1f} min",
                  flush=True)
            failures.append(key)
            if os.path.exists(out):
                os.remove(out)
            # Keep the worker files (run_batch retains them on failure) for
            # inspection and re-merge; a rerun of the same output stem
            # removes any stale work dir at startup.
            wip = temp_work_dir(output_stem(out))
            if os.path.isdir(wip):
                print(f"[warn] worker files kept in {wip} for inspection "
                      f"and re-merge", flush=True)

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
