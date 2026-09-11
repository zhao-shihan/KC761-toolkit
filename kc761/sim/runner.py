"""Run orchestration: memory-budgeted workers, uproot merging, product output.

Two scoring paths share this module:

* :func:`run_source` runs the radioactive-source mode and writes an
  ``mc_spectrum`` product (D-120);
* :func:`run_matrix` runs a matrix mode (plane/sphere) against a calibration
  product and writes a ``SimProduct`` (D-121).

Workers write only minimal raw ROOT histograms; merging happens here in Python
with ``uproot`` (D-11/D-125), followed by the physical-layer certificates
(D-127) and an atomic ``schema.write_product``. Randomness is derived per column
(matrix) and per event block (source) by F-SIM-7 (D-123), so a fixed seed and
worker partition reproduce bit-for-bit.

Geant4 is imported lazily inside the worker/interactive functions, so merging,
thread estimation and product assembly stay testable without Geant4.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import uproot

from kc761.core.binning import (
    MAX_CHANNELS,
    SOURCE_MODE_DEPOSITION_BINS,
    source_mode_deposition_edges_kev,
)
from kc761.errors import UsageError, ValidationError
from kc761.runtime import configure_logging
from kc761.schema.axes import (
    DEPOSITION_AXIS_NAME,
    ENERGY_AXIS_NAME,
    PRIMARY_AXIS_NAME,
    energy_axis,
)
from kc761.schema.io import (
    build_provenance,
    read_product,
    validate_output_path,
    write_product,
)
from kc761.schema.products import (
    SCHEMA_VERSION,
    CalibProduct,
    Histogram1D,
    Histogram2D,
    McSpectrumProduct,
    SimProduct,
)
from kc761.sim import certificates
from kc761.sim.config import (
    BYTES_PER_FLOAT64,
    DEFAULT_SEED,
    G4_WORKER_BASELINE_BYTES,
    MATRIX_HIST_NAME,
    MEMORY_SAFETY_FRACTION,
    SOURCE_MODE_EVENT_BLOCK,
    SOURCE_MODE_NAME,
    SPECTRUM_HIST_NAME,
    ZERO_DEPOSITION_HIST_NAME,
)
from kc761.sim.detector import build_plane_gamma_source, build_sphere_gamma_source
from kc761.sim.sources import (
    MODE_NAME_PLANE,
    MODE_NAME_SPHERE,
    MODE_PLANE,
    MODE_SPHERE,
    ColumnSchedule,
    ColumnSlice,
    MatrixSource,
    PrimaryAxis,
    SourceSpec,
    get_source,
    make_primary_axis,
    mode_metadata,
)

WINDOW_CHUNK = 4096
"""Extra float64 bins reserved for the spectrum term of the memory budget."""

_GEANT4_DEPENDENCY = "geant4_pybind"


def _logger() -> logging.Logger:
    return configure_logging("sim")


# --------------------------------------------------------------------------
# Memory budget (D-124)
# --------------------------------------------------------------------------
def available_memory_bytes() -> int | None:
    """Return the available system memory in bytes, or ``None`` if unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (OSError, ValueError):
        pass
    return None


def estimate_threads(
    n_deposition: int,
    n_primary: int,
    *,
    explicit: int | None = None,
    source_mode: bool = False,
) -> int:
    """Estimate the worker count from available memory and the per-worker cost.

    The per-worker footprint is the raw histogram storage (contents and
    ``fSumw2``) plus the documented Geant4 baseline; the estimate is capped by
    the CPU count and by the memory budget. An explicit positive ``threads``
    value overrides the estimate (D-124).
    """
    if explicit is not None:
        if explicit < 1:
            raise UsageError(f"threads must be >= 1, got {explicit!r}")
        return explicit
    cpu = os.cpu_count() or 1
    if source_mode:
        histogram_bytes = SOURCE_MODE_DEPOSITION_BINS * 2 * BYTES_PER_FLOAT64
    else:
        histogram_bytes = (
            n_deposition * n_primary * 2 + n_primary * 2
        ) * BYTES_PER_FLOAT64
        histogram_bytes += WINDOW_CHUNK * 2 * BYTES_PER_FLOAT64
    per_worker = histogram_bytes + G4_WORKER_BASELINE_BYTES
    available = available_memory_bytes()
    if available is None:
        reason = "available memory unknown; using the CPU count"
        chosen = cpu
    else:
        budget = int(available * MEMORY_SAFETY_FRACTION)
        memory_threads = max(1, budget // per_worker)
        chosen = min(cpu, memory_threads)
        reason = (
            f"available {available / 2**20:.0f} MiB, budget {budget / 2**20:.0f} MiB, "
            f"per worker {per_worker / 2**20:.0f} MiB, cpu {cpu}"
        )
    _logger().info("using %d worker(s): %s", chosen, reason)
    return max(1, chosen)


# --------------------------------------------------------------------------
# Raw histogram merging (D-125)
# --------------------------------------------------------------------------
def _axis_edges(hist, index: int) -> np.ndarray:  # noqa: ANN001
    return np.asarray(hist.axis(index).edges(), dtype=np.float64)


def merge_matrix_worker_histograms(
    paths: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum worker ``G`` and zero-deposition histograms with bitwise axis checks.

    Returns ``(counts, deposition_edges_kev, primary_edges_kev, zero_counts)``.
    """
    if not paths:
        raise ValidationError("no worker files to merge")
    counts_total: np.ndarray | None = None
    zero_total: np.ndarray | None = None
    ref_deposition: np.ndarray | None = None
    ref_primary: np.ndarray | None = None
    for path in paths:
        with uproot.open(path) as handle:
            if MATRIX_HIST_NAME not in handle or ZERO_DEPOSITION_HIST_NAME not in handle:
                raise ValidationError(
                    f"worker file {path!r} is missing the matrix histograms"
                )
            matrix = handle[MATRIX_HIST_NAME]
            zero = handle[ZERO_DEPOSITION_HIST_NAME]
            deposition_edges = _axis_edges(matrix, 0)
            primary_edges = _axis_edges(matrix, 1)
            zero_edges = _axis_edges(zero, 0)
            if not np.array_equal(primary_edges, zero_edges):
                raise ValidationError(
                    f"worker file {path!r}: zero-deposition axis does not match G.y"
                )
            if ref_deposition is None:
                ref_deposition = deposition_edges
                ref_primary = primary_edges
            else:
                if not np.array_equal(deposition_edges, ref_deposition):
                    raise ValidationError(
                        f"worker file {path!r}: deposition edges differ between workers"
                    )
                if not np.array_equal(primary_edges, ref_primary):
                    raise ValidationError(
                        f"worker file {path!r}: primary edges differ between workers"
                    )
            values = np.asarray(matrix.values(), dtype=np.float64)
            zero_values = np.asarray(zero.values(), dtype=np.float64)
            if values.shape != (deposition_edges.size - 1, primary_edges.size - 1):
                raise ValidationError(
                    f"worker file {path!r}: G shape {values.shape} does not match its axes"
                )
            if zero_values.shape != (primary_edges.size - 1,):
                raise ValidationError(
                    f"worker file {path!r}: zero-deposition shape {zero_values.shape} "
                    "does not match the primary axis"
                )
            counts_total = values if counts_total is None else counts_total + values
            zero_total = zero_values if zero_total is None else zero_total + zero_values
    assert counts_total is not None and zero_total is not None
    assert ref_deposition is not None and ref_primary is not None
    return counts_total, ref_deposition, ref_primary, zero_total


def merge_source_worker_histograms(
    paths: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sum worker pulse spectra with a bitwise energy-axis check."""
    if not paths:
        raise ValidationError("no worker files to merge")
    total: np.ndarray | None = None
    reference: np.ndarray | None = None
    for path in paths:
        with uproot.open(path) as handle:
            if SPECTRUM_HIST_NAME not in handle:
                raise ValidationError(
                    f"worker file {path!r} is missing histogram {SPECTRUM_HIST_NAME!r}"
                )
            hist = handle[SPECTRUM_HIST_NAME]
            edges = _axis_edges(hist, 0)
            values = np.asarray(hist.values(), dtype=np.float64)
            if values.shape != (edges.size - 1,):
                raise ValidationError(
                    f"worker file {path!r}: spectrum shape {values.shape} does not "
                    "match its axis"
                )
            if reference is None:
                reference = edges
            elif not np.array_equal(edges, reference):
                raise ValidationError(
                    f"worker file {path!r}: spectrum energy edges differ between workers"
                )
            total = values if total is None else total + values
    assert total is not None and reference is not None
    return total, reference


# --------------------------------------------------------------------------
# Worker process functions (Geant4 imported lazily)
# --------------------------------------------------------------------------
def _build_serial_manager(detector, action_init, seed: int, verbose: int):  # noqa: ANN001, ANN202
    from geant4_pybind import G4Random, G4RunManagerFactory, G4RunManagerType

    G4Random.setTheSeed(int(seed))
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        run_manager = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    run_manager.SetUserInitialization(detector)
    from kc761.sim.physics import build_physics_list

    run_manager.SetUserInitialization(build_physics_list())
    run_manager.SetUserInitialization(action_init)
    _apply_verbosity(run_manager, verbose)
    run_manager.SetPrintProgress(5000)
    run_manager.Initialize()
    return run_manager


def _apply_verbosity(run_manager, verbose: int) -> None:  # noqa: ANN001
    from geant4_pybind import G4EmParameters, G4HadronicParameters, G4UImanager

    run_manager.SetVerboseLevel(verbose)
    G4HadronicParameters.Instance().SetVerboseLevel(verbose)
    G4EmParameters.Instance().SetVerbose(verbose)
    G4UImanager.GetUIpointer().ApplyCommand(f"/run/verbose {verbose}")


def source_worker(
    source_key: str,
    output_stem: str,
    n_events: int,
    seed: int,
    event_offset: int,
    verbose: int,
) -> int:
    """Run one source-mode worker and return its event count."""
    from kc761.sim import actions, detector, materials, physics

    spec = get_source(source_key)
    materials_map = materials.build_all_materials(spec)
    construction = detector.build_detector(
        spec, materials_map, check_overlaps=verbose > 0
    )
    action_init = actions.build_source_action_initialization(
        spec, construction, output_stem, event_offset, seed, verbose
    )
    run_manager = _build_serial_manager(construction, action_init, seed, verbose)
    physics.configure_radioactive_decay(spec)
    physics.configure_gps(spec, construction)
    run_manager.BeamOn(n_events)
    return n_events


def matrix_worker(
    output_stem: str,
    column_slice: ColumnSlice,
    axis: PrimaryAxis,
    source: MatrixSource,
    deposition_edges_kev: np.ndarray,
    seed: int,
    verbose: int,
) -> int:
    """Run one matrix-mode worker over its column slice."""
    from kc761.sim import actions, detector, materials

    materials_map = materials.build_all_materials()
    construction = detector.build_detector(
        None, materials_map, check_overlaps=verbose > 0
    )
    action_init = actions.build_matrix_action_initialization(
        column_slice,
        axis,
        source,
        construction,
        output_stem,
        deposition_edges_kev,
        seed,
        verbose,
    )
    run_manager = _build_serial_manager(construction, action_init, seed, verbose)
    run_manager.BeamOn(column_slice.total_events)
    return column_slice.total_events


# --------------------------------------------------------------------------
# Batch orchestration
# --------------------------------------------------------------------------
def _worker_directory(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.stem}-wip-", dir=output.parent))


def _run_pool(
    worker: Callable[..., int], arguments: Sequence[tuple], threads: int
) -> None:
    """Run every worker in its own spawned process.

    Geant4 forbids constructing a second run manager in one process
    (``G4RunManager`` aborts), so even a single worker runs in a subprocess.
    ``spawn`` also isolates the Geant4 global state between workers and callers.
    """
    if not arguments:
        return
    processes = max(1, min(threads, len(arguments)))
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=processes) as pool:
        pool.starmap(worker, arguments)


def _write_matrix_product(
    counts: np.ndarray,
    deposition_edges_kev: np.ndarray,
    primary_edges_kev: np.ndarray,
    zero_counts: np.ndarray,
    *,
    source: MatrixSource,
    n_events: int,
    seed: int,
    workers: int,
    output: Path,
    calib_path: Path,
    force: bool,
    strict: bool,
    command: str,
    arguments: Sequence[tuple[str, str]],
    extra_inputs: Sequence[str | Path] = (),
) -> Path:
    totals = counts.sum(axis=0) + zero_counts
    certificates.verify_event_accounting(counts, totals, zero_counts, n_events)
    certificates.verify_binomial_variance(counts, totals)
    certificates.verify_efficiency(counts, totals)
    certificates.verify_physical_boundary(
        counts, deposition_edges_kev, primary_edges_kev, strict=strict
    )
    if counts.shape != (deposition_edges_kev.size - 1, primary_edges_kev.size - 1):
        raise ValidationError("merged G shape does not match its axes")
    mode, mode_name, geometry_name, geometry_param_mm = mode_metadata(source)
    provenance = build_provenance(
        producer="kc761-sim",
        command=command,
        arguments=tuple(arguments),
        inputs=[calib_path, *extra_inputs],
        extra_dependencies=[_GEANT4_DEPENDENCY],
    )
    product = SimProduct(
        format_version=SCHEMA_VERSION,
        primary_to_deposition=Histogram2D(
            x=energy_axis(deposition_edges_kev, name=DEPOSITION_AXIS_NAME),
            y=energy_axis(primary_edges_kev, name=PRIMARY_AXIS_NAME),
            values=counts,
            variances=certificates.binomial_variance(counts, totals),
        ),
        primary_column_totals=Histogram1D(
            axis=energy_axis(primary_edges_kev, name=PRIMARY_AXIS_NAME),
            values=totals,
        ),
        mode=mode,
        mode_name=mode_name,
        geometry_name=geometry_name,
        geometry_param_mm=geometry_param_mm,
        angular_distribution=source.angular,
        seed=seed,
        n_events=n_events,
        workers=workers,
        provenance=provenance,
    )
    return write_product(product, output, force=force, strict=strict)


def run_matrix(
    mode: str | int,
    calib: str | Path,
    output: str | Path,
    *,
    n_events: int,
    seed: int = DEFAULT_SEED,
    threads: int | None = None,
    force: bool = False,
    strict: bool = False,
    verbose: int = 0,
    command: str = "kc761 sim",
    arguments: Sequence[tuple[str, str]] = (),
    extra_inputs: Sequence[str | Path] = (),
) -> Path:
    """Run a matrix-mode simulation and write a ``SimProduct`` (D-121).

    The calibration product supplies both energy axes: the deposition axis is
    ``C.y`` and the primary (incident gamma) axis is the *same* channel-derived
    axis, so ``G`` is a square matrix on the legacy layout. This is the frozen
    D-121 revision; the fixed source-mode Monte-Carlo axis is used only by the
    source-mode ``mc_spectrum``.
    """
    validate_output_path(output, force=force)
    if n_events <= 0:
        raise UsageError(f"n_events must be positive, got {n_events!r}")
    source = _matrix_source(mode)
    calib_path = Path(calib)
    product = read_product(calib_path, strict=strict)
    if not isinstance(product, CalibProduct):
        raise ValidationError(f"{calib_path}: expected a calib product")
    deposition_edges = np.asarray(product.deposition_to_channel.y.edges, dtype=np.float64)
    n_deposition = int(deposition_edges.size) - 1
    if n_deposition > MAX_CHANNELS:
        raise ValidationError(
            f"calibration deposition axis has {n_deposition} bins, above the "
            f"supported maximum {MAX_CHANNELS} (D-52)"
        )
    axis = make_primary_axis(deposition_edges)
    schedule = ColumnSchedule.fixed_total(axis, n_events)
    workers = estimate_threads(
        deposition_edges.size - 1,
        axis.n_columns,
        explicit=threads,
    )
    slices = tuple(
        chunk for chunk in schedule.slices(workers) if chunk.total_events > 0
    )
    output_path = Path(output)
    work_dir = _worker_directory(output_path)
    try:
        stems = [str(work_dir / f"w{index}") for index in range(len(slices))]
        full = [
            (stems[index], slices[index], axis, source, deposition_edges, seed, verbose)
            for index in range(len(slices))
        ]
        _run_pool(matrix_worker, full, workers)
        paths = [stem + ".root" for stem in stems]
        counts, dep_edges, primary_edges, zero_counts = merge_matrix_worker_histograms(paths)
        return _write_matrix_product(
            counts,
            dep_edges,
            primary_edges,
            zero_counts,
            source=source,
            n_events=n_events,
            seed=seed,
            workers=len(paths),
            output=output_path,
            calib_path=calib_path,
            force=force,
            strict=strict,
            command=command,
            arguments=arguments,
            extra_inputs=extra_inputs,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _matrix_source(mode: str | int) -> MatrixSource:
    # Canonical hyphen tokens only (D-143/D-2); no underscore or short aliases.
    if mode in (MODE_PLANE, MODE_NAME_PLANE):
        return build_plane_gamma_source()
    if mode in (MODE_SPHERE, MODE_NAME_SPHERE):
        return build_sphere_gamma_source()
    raise UsageError(
        f"unknown matrix mode {mode!r}; expected {MODE_NAME_PLANE} or {MODE_NAME_SPHERE}"
    )


def _write_mc_spectrum(
    values: np.ndarray,
    energy_edges_kev: np.ndarray,
    *,
    spec: SourceSpec,
    n_events: int,
    seed: int,
    workers: int,
    output: Path,
    force: bool,
    strict: bool,
    command: str,
    arguments: Sequence[tuple[str, str]],
    extra_inputs: Sequence[str | Path] = (),
) -> Path:
    expected = source_mode_deposition_edges_kev()
    if not np.array_equal(energy_edges_kev, expected):
        raise ValidationError(
            "source-mode spectrum axis does not match the fixed source-mode "
            "deposition axis"
        )
    variances = certificates.source_spectrum_variance(values)
    certificates.verify_source_spectrum_variance(values, variances)
    provenance = build_provenance(
        producer="kc761-sim",
        command=command,
        arguments=tuple(arguments),
        inputs=list(extra_inputs),
        extra_dependencies=[_GEANT4_DEPENDENCY],
    )
    product = McSpectrumProduct(
        format_version=SCHEMA_VERSION,
        spectrum=Histogram1D(
            axis=energy_axis(energy_edges_kev, name=ENERGY_AXIS_NAME),
            values=values,
            variances=variances,
        ),
        source_key=spec.key,
        mode_name=SOURCE_MODE_NAME,
        geometry_name=spec.geometry_name,
        geometry_param_mm=spec.geometry_param_mm,
        n_events=n_events,
        seed=seed,
        workers=workers,
        provenance=provenance,
    )
    return write_product(product, output, force=force, strict=strict)


def run_source(
    source_key: str,
    output: str | Path,
    *,
    n_events: int,
    seed: int = DEFAULT_SEED,
    threads: int | None = None,
    force: bool = False,
    strict: bool = False,
    verbose: int = 0,
    command: str = "kc761 sim",
    arguments: Sequence[tuple[str, str]] = (),
    extra_inputs: Sequence[str | Path] = (),
) -> Path:
    """Run the radioactive-source mode and write an ``mc_spectrum`` (D-120)."""
    validate_output_path(output, force=force)
    if n_events <= 0:
        raise UsageError(f"n_events must be positive, got {n_events!r}")
    spec = get_source(source_key)
    workers = estimate_threads(0, 0, explicit=threads, source_mode=True)
    block = SOURCE_MODE_EVENT_BLOCK
    n_blocks = (n_events + block - 1) // block
    workers = min(workers, n_blocks)
    # Partition whole blocks contiguously so each block's stream is fixed.
    base, remainder = divmod(n_blocks, workers)
    chunks: list[tuple[int, int, int]] = []
    event_start = 0
    for index in range(workers):
        n_worker_blocks = base + (1 if index < remainder else 0)
        worker_events = min(n_worker_blocks * block, n_events - event_start)
        chunks.append((worker_events, event_start, index))
        event_start += worker_events
    output_path = Path(output)
    work_dir = _worker_directory(output_path)
    try:
        stems = [str(work_dir / f"w{index}") for index in range(len(chunks))]
        arguments_list = [
            (source_key, stems[index], count, seed, offset, verbose)
            for index, (count, offset, _) in enumerate(chunks)
        ]
        _run_pool(source_worker, arguments_list, len(arguments_list))
        paths = [stem + ".root" for stem in stems]
        values, energy_edges = merge_source_worker_histograms(paths)
        return _write_mc_spectrum(
            values,
            energy_edges,
            spec=spec,
            n_events=n_events,
            seed=seed,
            workers=len(paths),
            output=output_path,
            force=force,
            strict=strict,
            command=command,
            arguments=arguments,
            extra_inputs=extra_inputs,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def prepare_interactive(
    source_key: str,
    *,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
):  # noqa: ANN201
    """Build an initialized source-mode run manager for a UI session (D-126).

    The caller (W6) drives the Geant4 UI, executes ``init_vis.mac`` and
    ``vis.mac`` and calls ``/run/beamOn``. No output file is opened until a run
    starts.
    """
    from kc761.sim import actions, detector, materials, physics

    spec = get_source(source_key)
    materials_map = materials.build_all_materials(spec)
    construction = detector.build_detector(spec, materials_map, check_overlaps=verbose > 0)
    action_init = actions.build_source_action_initialization(
        spec, construction, "interactive", 0, seed, verbose
    )
    run_manager = _build_serial_manager(construction, action_init, seed, verbose)
    physics.configure_radioactive_decay(spec)
    physics.configure_gps(spec, construction)
    return run_manager


def macro_path(name: str) -> Path:
    """Return the absolute path of a simulation macro (D-126)."""
    if name not in {"vis.mac", "init_vis.mac", "gui.mac"}:
        raise UsageError(f"unknown simulation macro {name!r}")
    return Path(__file__).with_name(name)


__all__ = [
    "available_memory_bytes",
    "estimate_threads",
    "macro_path",
    "matrix_worker",
    "merge_matrix_worker_histograms",
    "merge_source_worker_histograms",
    "prepare_interactive",
    "run_matrix",
    "run_source",
    "source_worker",
]
