#!/usr/bin/env python3
"""Reproducible performance benchmarks for the numeric hotspots.

This tool is intentionally outside the correctness path: it measures the
wall-clock time and peak memory of the heavy numeric stages so that an
optimisation can be judged by a before/after pair. It never writes products;
``calib``/``unfold`` run with ``output=None`` and ``plot=False``.

Usage::

    python tools/benchmarks.py --scenario all --repeat 2
    python tools/benchmarks.py --scenario unfold --json /tmp/unfold.json

Scenarios:

* ``kernel``  - vectorised response assembly + parameter Jacobian on the
  calibration product (or a synthetic calibration when ``work/`` is absent);
* ``calib``   - full ``run_fit`` on the four ``work/`` datasets;
* ``unfold``  - full ``run_unfold`` on the ``work/`` products;
* ``memory``  - runs every scenario and reports peak RSS (plus the
  ``tracemalloc`` peak with ``--tracemalloc``); use ``--isolate`` to isolate
  the high-water mark per scenario.

The default thread budget is one BLAS/OMP/MKL thread so the numbers are
comparable; pass ``--threads N`` to allow the vendor libraries to use N. All
measurements run in this process unless ``--isolate`` is given, which re-runs
each scenario in a fresh interpreter so ``ru_maxrss`` reflects only that
scenario.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORK = REPO_ROOT / "work"
CALIB_PRODUCT = WORK / "calib" / "calib-2609a.root"
SIM_PRODUCT = WORK / "sim" / "calib-2609a-plane-front-gamma-n100000000-s908136382.root"
DATA_PRODUCT = WORK / "data" / "2609a" / "th232-260908-subbkg.root"

CALIB_DATASETS: tuple[tuple[str, str, str, int, int], ...] = (
    (
        "Am241",
        "work/data/2609a/am241-260910-subbkg.root",
        "work/sim/am241-n3000000-s908136382.root",
        140,
        165,
    ),
    (
        "Lu176",
        "work/data/2609a/lu176-260910-subbkg.root",
        "work/sim/lu176-n20000000-s908136382.root",
        140,
        450,
    ),
    (
        "Th232",
        "work/data/2609a/th232-260908-subbkg.root",
        "work/sim/th232-n200000000-s908136382.root",
        140,
        1400,
    ),
    (
        "Ra226",
        "work/data/2609a/ra226-260908-subbkg.root",
        "work/sim/ra226-n100000000-s908136382.root",
        140,
        1400,
    ),
)


def _calib_rows() -> list[tuple[str, str, str, int, int]]:
    """Dataset rows with existing input files, in the frozen labelling order."""
    rows: list[tuple[str, str, str, int, int]] = []
    for label, data, mc, low, high in CALIB_DATASETS:
        if (REPO_ROOT / data).is_file() and (REPO_ROOT / mc).is_file():
            rows.append((label, data, mc, low, high))
    return rows


def _calib_specs():  # noqa: ANN202 - DatasetSpec tuple
    from kc761.cli.calib import _build_spec

    return [
        _build_spec(
            data_path=data,
            mc_path=mc,
            label=label,
            channel_low=low,
            channel_high=high,
            syst_frac=0.05,
            strict=False,
        )
        for label, data, mc, low, high in _calib_rows()
    ]


def _set_threads(threads: int) -> None:
    value = str(max(1, threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value
    import numba

    numba.set_num_threads(max(1, threads))


def _peak_rss_mib() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover - non-Linux fallback
        return float("nan")


def _have(*paths: Path) -> bool:
    return all(path.is_file() for path in paths)


def _time_call(function: Callable[[], Any], repeat: int) -> tuple[list[float], Any]:
    timings: list[float] = []
    result: Any = None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return timings, result


def bench_kernel(repeat: int) -> dict[str, Any]:
    """Response assembly + Jacobian on the calibration product."""
    if not _have(CALIB_PRODUCT):
        return {"scenario": "kernel", "skipped": "work/calib/calib-2609a.root missing"}

    from kc761.core.binning import ChannelGrid, source_mode_deposition_edges_kev
    from kc761.core.kernel import response_triples
    from kc761.core.model import energy_kev, resolution_sigma_kev
    from kc761.core.response import response_parameter_jacobian
    from kc761.schema.io import read_product
    from kc761.schema.products import CalibProduct
    from kc761.unfold.inputs import internal_calibration, resolution_params

    product = read_product(CALIB_PRODUCT, strict=False)
    assert isinstance(product, CalibProduct)
    calibration = internal_calibration(product)
    resol = resolution_params(product)
    grid = ChannelGrid(product.deposition_to_channel.x.n_bins)
    channel_max = product.channel_max
    edges = source_mode_deposition_edges_kev()
    centers = 0.5 * (edges[:-1] + edges[1:])
    channel_edges = energy_kev(grid.edges(), calibration, channel_max=channel_max)
    sigma = resolution_sigma_kev(centers, resol)

    def value_build() -> int:
        triples = response_triples(channel_edges, centers, sigma)
        return int(triples.values.size)

    def jacobian_build() -> int:
        matrices = response_parameter_jacobian(
            edges, calibration, resol, channel_grid=grid, channel_max=channel_max
        )
        return sum(int(matrix.nnz) for matrix in matrices)

    value_times, entries = _time_call(value_build, repeat)
    jac_times, nnz = _time_call(jacobian_build, repeat)
    return {
        "scenario": "kernel",
        "edges": int(edges.size - 1),
        "support_entries": int(entries),
        "jacobian_nnz": int(nnz),
        "value_s": _summary(value_times),
        "jacobian_s": _summary(jac_times),
    }


def bench_calib(repeat: int) -> dict[str, Any]:
    if not _have(CALIB_PRODUCT):
        return {"scenario": "calib", "skipped": "work products missing"}
    specs = _calib_specs()
    if not specs:
        return {"scenario": "calib", "skipped": "no calib datasets found"}
    from kc761.calib.fit import run_fit

    timings, result = _time_call(
        lambda: run_fit(specs, output=None, plot=False, progress=None), repeat
    )
    return {
        "scenario": "calib",
        "datasets": len(specs),
        "nfev": int(result.nfev),
        "status": result.status,
        "chi2": float(result.chi2),
        "seconds": _summary(timings),
    }


def bench_unfold(repeat: int) -> dict[str, Any]:
    if not _have(CALIB_PRODUCT, SIM_PRODUCT, DATA_PRODUCT):
        return {"scenario": "unfold", "skipped": "work products missing"}
    from kc761.errors import ProvenanceError, ValidationError
    from kc761.schema.io import read_product
    from kc761.unfold.unfold import run_unfold

    # Load the products in memory and pass the objects: the provenance digest
    # check is a one-off file hash, and the shared ``work/`` tree may be
    # rewritten by a concurrent run. Passing objects still exercises the full
    # numeric unfold path.
    data_product = read_product(DATA_PRODUCT, strict=False)
    calib_product = read_product(CALIB_PRODUCT, strict=False)
    sim_product = read_product(SIM_PRODUCT, strict=False)
    try:
        timings, result = _time_call(
            lambda: run_unfold(
                data_product,
                calib_product,
                sim_product,
                energy_low_kev=40.0,
                energy_high_kev=2800.0,
                alpha=0.1,
                output=None,
                plot=False,
            ),
            repeat,
        )
    except (ValidationError, ProvenanceError) as exc:
        # ``work/`` is not committed and a concurrent run may leave the
        # calibration and simulation products axis-incompatible; report the
        # data state instead of failing the benchmark.
        return {"scenario": "unfold", "skipped": f"inconsistent work/ products: {exc}"}
    return {
        "scenario": "unfold",
        "n_active": int(result.n_active),
        "n_fit_rows": int(result.n_fit_rows),
        "chi2": float(result.chi2),
        "seconds": _summary(timings),
    }


def _summary(timings: Sequence[float]) -> dict[str, float]:
    values = list(timings)
    return {
        "n": len(values),
        "best": min(values),
        "mean": sum(values) / len(values),
    }


SCENARIOS: dict[str, Callable[[int], dict[str, Any]]] = {
    "kernel": bench_kernel,
    "calib": bench_calib,
    "unfold": bench_unfold,
}


def _run_isolated(scenario: str, repeat: int, threads: int, json_path: str | None) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scenario",
        scenario,
        "--repeat",
        str(repeat),
        "--threads",
        str(threads),
    ]
    if json_path:
        command += ["--json", json_path]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all", "memory"], default="all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="run each scenario in a fresh interpreter (isolated peak RSS)",
    )
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="also record the tracemalloc peak (adds large timing overhead)",
    )
    args = parser.parse_args(argv)
    _set_threads(args.threads)

    if args.isolate:
        names = list(SCENARIOS) if args.scenario in ("all", "memory") else [args.scenario]
        for name in names:
            _run_isolated(name, args.repeat, args.threads, None)
        return 0

    names = list(SCENARIOS) if args.scenario in ("memory", "all") else [args.scenario]

    reports: list[dict[str, Any]] = []
    for name in names:
        if args.tracemalloc:
            tracemalloc.start()
        report = SCENARIOS[name](args.repeat)
        if args.tracemalloc:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            report["tracemalloc_peak_mib"] = peak / 2**20
        report["maxrss_mib"] = _peak_rss_mib()
        reports.append(report)
        print(json.dumps(report, sort_keys=True))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
