"""Run orchestration: single-process simulations and multiprocessing batching.

Two scoring paths share the run-manager assembly and the multiprocessing
batch machinery but differ in their outputs:

* the radioactive-source path writes the ntuple + spectrum histogram;
* the matrix-mode path writes only the primary-to-deposition histogram (TH2D)
  and the zero-deposition counter (TH1D), which are merged with hadd and
  exported as the simulation file by :mod:`kc761sim.export` (the primary-to-channel
  composition happens at unfold time).

The merge validation is driven by a :class:`MergeExpectation` descriptor,
so both paths share :func:`merge_worker_outputs` without hardcoding either
schema.
"""

from __future__ import annotations

import contextlib
import functools
import multiprocessing
import os
import shutil
from dataclasses import dataclass

from geant4_pybind import (
    G4EmParameters,
    G4HadronicParameters,
    G4Random,
    G4RunManager,
    G4RunManagerFactory,
    G4RunManagerType,
    G4UImanager,
)
from . import (
    actions,
    config,
    detector,
    materials,
    physics,
)
from .config import SourceSpec
from .paths import (
    PRIMARY_TO_DEPOSITION_HIST_NAME,
    ZERO_DEPOSITION_HIST_NAME,
    NTUPLE_NAME,
    SPECTRUM_HIST_NAME,
    final_output_path,
    output_stem,
    temp_work_dir,
)
from .sources import MatrixSource

DEFAULT_SEED = 908136382


@dataclass(frozen=True)
class MergeExpectation:
    """Objects a merged worker output must contain, and how to validate them.

    ``hist_names``: histograms (TH1/TH2) summed by hadd whose binning must
    be identical across the worker files, checked axis by axis before
    merging (hadd itself silently adds differently binned histograms
    bin-by-bin, which would corrupt the result).
    ``tree_names``: ntuples whose entry counts must sum across the workers
    (the merged tree then carries that exact entry count).
    """

    hist_names: tuple[str, ...] = ()
    tree_names: tuple[str, ...] = ()

    @classmethod
    def radioactive(cls) -> "MergeExpectation":
        return cls(hist_names=(SPECTRUM_HIST_NAME,),
                   tree_names=(NTUPLE_NAME,))

    @classmethod
    def matrix(cls) -> "MergeExpectation":
        return cls(hist_names=(PRIMARY_TO_DEPOSITION_HIST_NAME, ZERO_DEPOSITION_HIST_NAME))


def apply_verbosity(run_manager: G4RunManager, verbose: int) -> None:
    run_manager.SetVerboseLevel(verbose)
    G4HadronicParameters.Instance().SetVerboseLevel(verbose)
    G4EmParameters.Instance().SetVerbose(verbose)
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(f"/run/verbose {verbose}")


def _build_serial_run_manager(
    det,
    action_init,
    seed: int,
    verbose: int,
) -> G4RunManager:
    """Assemble and initialize the serial run manager shared by both paths.

    Both scoring paths build the same manager (seed, detector, physics
    list, action set, verbosity, progress reporting, Initialize); only
    the detector/action-set arguments and the optional post-init
    configuration differ.
    """
    G4Random.setTheSeed(int(seed))

    # Silence pybind11's static-initialization chatter on first import.
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            run_manager = G4RunManagerFactory.CreateRunManager(
                G4RunManagerType.Serial)

    run_manager.SetUserInitialization(det)
    run_manager.SetUserInitialization(physics.PhysicsList())
    run_manager.SetUserInitialization(action_init)
    apply_verbosity(run_manager, verbose)
    run_manager.SetPrintProgress(5000)
    run_manager.Initialize()
    return run_manager


def prepare_run_manager(
    spec: SourceSpec,
    output_stem: str,
    seed: int,
    event_offset: int = 0,
    verbose: int = 0,
) -> G4RunManager:
    """Assemble, initialize and configure a serial run manager.

    Shared by batch workers and the interactive session; returns the
    manager ready for ``BeamOn`` (batch) or manual /run/beamOn (UI).
    """
    mats = materials.build_all_materials(spec)
    det = detector.DetectorConstruction(
        spec, mats, check_overlaps=verbose > 0)
    run_manager = _build_serial_run_manager(
        det,
        actions.ActionInitialization(
            spec, det, output_stem, event_offset, verbose),
        seed, verbose)
    physics.configure_radioactive_decay(spec)
    physics.configure_gps(spec, det)
    return run_manager


def prepare_matrix_run_manager(
    source: MatrixSource,
    output_stem: str,
    seed: int,
    event_offset: int = 0,
    verbose: int = 0,
) -> G4RunManager:
    """Assemble, initialize and configure a serial matrix-mode run manager.

    Same shape as :func:`prepare_run_manager` but for the bare detector
    (no source volumes), the surface generator and the G scoring actions;
    no GPS or radioactive-decay configuration is applied.
    """
    mats = materials.build_all_materials()
    det = detector.DetectorConstruction(
        None, mats, check_overlaps=verbose > 0)
    return _build_serial_run_manager(
        det,
        actions.MatrixActionInitialization(
            source, det, output_stem, event_offset, verbose),
        seed, verbose)


def run_simulation(
    source_key: str,
    output_stem: str,
    n_events: int,
    seed: int,
    event_offset: int = 0,
    verbose: int = 0,
) -> int:
    """Run one single-process simulation; returns the number of events."""
    spec = config.SOURCES[source_key]
    run_manager = prepare_run_manager(
        spec, output_stem, seed=seed,
        event_offset=event_offset, verbose=verbose)
    run_manager.BeamOn(n_events)
    return n_events


def run_matrix_simulation(
    source: MatrixSource,
    output_stem: str,
    n_events: int,
    seed: int,
    event_offset: int = 0,
    verbose: int = 0,
) -> int:
    """Run one single-process matrix-mode simulation; returns the event count."""
    run_manager = prepare_matrix_run_manager(
        source, output_stem, seed=seed,
        event_offset=event_offset, verbose=verbose)
    run_manager.BeamOn(n_events)
    return n_events


def _split_events(n_events: int, n_parts: int) -> list[int]:
    base, remainder = divmod(n_events, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _validate_merged_output(output_path: str, expected_entries: int,
                            expectation: MergeExpectation) -> None:
    """Check the hadd output and remove it when it is incomplete.

    A failed or aborted merge can leave a truncated-but-readable file at
    the production path; later runs would skip it as an existing output.
    Require every object named by ``expectation`` to be present, the
    ntuple entry count to match the sum of the worker entries, and (for
    histograms) the binning to match the worker files (already enforced
    per worker before merging).
    """
    import uproot

    reason = None
    try:
        with uproot.open(output_path) as f:
            for name in expectation.tree_names:
                if name not in f:
                    reason = f"missing ntuple {name!r}"
                    break
                n_entries = int(f[name].num_entries)
                if n_entries != expected_entries:
                    reason = (f"{n_entries} ntuple entries in {name!r}, "
                              f"expected {expected_entries}")
                    break
            else:
                for name in expectation.hist_names:
                    if name not in f:
                        reason = f"missing histogram {name!r}"
                        break
    except Exception as exc:
        reason = f"unreadable output: {exc}"
    if reason is not None:
        _remove_file(output_path)
        raise RuntimeError(
            f"hadd output {output_path!r} failed validation ({reason})")


def merge_worker_outputs(output_path: str, input_paths: list[str], *,
                         expectation: MergeExpectation,
                         hadd_exe: str | None = None) -> int:
    """Merge worker ROOT files into the final simulation output via hadd.

    Delegates to :func:`kc761util.hadd.merge_root_files` (ntuples merged
    entry-by-entry, histograms summed with their ``sumw2`` buffers).  The
    worker histogram binnings are validated axis by axis before merging
    (there the hadd bin-by-bin addition would corrupt the result): the
    objects to expect come from the ``expectation`` descriptor, which keeps
    the radioactive and matrix paths sharing this function.  After merging,
    the output is validated and a partial output from a failed merge is
    removed.
    """
    import numpy as np
    import uproot

    from kc761util.hadd import merge_root_files

    if not input_paths:
        raise ValueError("merge_worker_outputs: no worker files to merge")

    ref_edges: dict[str, list[np.ndarray]] = {}
    expected_entries = 0
    for path in input_paths:
        with uproot.open(path) as src:
            for name in expectation.tree_names:
                if name not in src:
                    raise RuntimeError(
                        f"ntuple {name!r} missing in worker file {path!r}")
                expected_entries += int(src[name].num_entries)
            for name in expectation.hist_names:
                try:
                    hist = src[name]
                except KeyError as exc:
                    raise RuntimeError(
                        f"histogram {name!r} missing in worker file "
                        f"{path!r}") from exc
                edges = [np.asarray(hist.axis(axis).edges(), dtype=float)
                         for axis in range(len(hist.axes))]
                if name not in ref_edges:
                    ref_edges[name] = edges
                elif any(not np.array_equal(e, r)
                         for e, r in zip(edges, ref_edges[name])):
                    raise RuntimeError(
                        f"histogram {name!r} bin edges differ between "
                        f"worker files ({path!r} disagrees with earlier "
                        f"workers)")

    rc = merge_root_files(output_path, input_paths, hadd_exe=hadd_exe)
    if rc != 0:
        _remove_file(output_path)
        return rc
    _validate_merged_output(output_path, expected_entries, expectation)
    return 0


def _run_batch(
    worker_fn,
    expectation: MergeExpectation,
    output_path: str,
    n_events: int,
    threads: int,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
    hadd_exe: str | None = None,
) -> None:
    """Run a simulation on ``threads`` workers and merge their outputs.

    ``worker_fn`` has the signature ``(output_stem, n_events, seed,
    event_offset, verbose) -> n_events`` (e.g. a
    :func:`functools.partial` of :func:`run_simulation` or
    :func:`run_matrix_simulation`); ``expectation`` describes the objects
    the merged output must contain.
    """
    stem = output_stem(output_path)
    final_path = final_output_path(output_path)
    # The Geant4 ROOT analysis manager appends ".root" only to file names
    # without a dot; a dotted stem would silently produce extension-less
    # worker files that the merge cannot find.
    if "." in os.path.basename(stem):
        raise ValueError(
            f"output stem {stem!r} must not contain a dot (the Geant4 "
            f"analysis manager only appends '.root' to dot-free names)")

    if threads <= 1:
        worker_fn(stem, n_events, seed, 0, verbose)
        return

    chunks = _split_events(n_events, threads)
    offsets = []
    running = 0
    for chunk in chunks:
        offsets.append(running)
        running += chunk

    work_dir = temp_work_dir(stem)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    base = os.path.basename(stem)

    worker_stems: list[str] = []
    tasks: list[multiprocessing.pool.AsyncResult] = []
    pool = multiprocessing.Pool(threads)
    try:
        try:
            for i, (chunk, offset) in enumerate(zip(chunks, offsets)):
                if chunk <= 0:
                    continue
                worker_stems.append(os.path.join(work_dir, f"{base}-w{i}"))
                tasks.append(
                    pool.apply_async(
                        worker_fn,
                        (
                            worker_stems[-1],
                            chunk,
                            seed + i + 1,
                            offset,
                            verbose,
                        ),
                    )
                )
            pool.close()
            pool.join()
            for task in tasks:
                task.get()
            rc = merge_worker_outputs(
                final_path, [s + ".root" for s in worker_stems],
                expectation=expectation, hadd_exe=hadd_exe)
            if rc != 0:
                raise RuntimeError(
                    f"hadd merge failed (exit code {rc}); worker files kept "
                    f"in {work_dir} for inspection")
        finally:
            pool.terminate()
    except BaseException:
        # Keep the worker files on any failure (including the hadd merge and
        # the binning check) so the run can be inspected and re-merged.
        raise
    shutil.rmtree(work_dir, ignore_errors=True)


def run_batch(
    source_key: str,
    output_path: str,
    n_events: int,
    threads: int,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
    hadd_exe: str | None = None,
) -> None:
    """Run a radioactive-source simulation on ``threads`` workers."""
    _run_batch(
        functools.partial(run_simulation, source_key),
        MergeExpectation.radioactive(),
        output_path, n_events, threads, seed, verbose, hadd_exe)


def run_batch_matrix(
    source: MatrixSource,
    output_path: str,
    n_events: int,
    threads: int,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
    hadd_exe: str | None = None,
) -> None:
    """Run a matrix-mode simulation on ``threads`` workers.

    The merged output contains only the G histogram and the zero-deposition
    counter; the primary-to-channel response matrix is composed from it
    afterwards by
    :mod:`kc761sim.export`.
    """
    _run_batch(
        functools.partial(run_matrix_simulation, source),
        MergeExpectation.matrix(),
        output_path, n_events, threads, seed, verbose, hadd_exe)
