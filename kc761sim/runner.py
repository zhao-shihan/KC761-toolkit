"""Run orchestration: single-process simulations and multiprocessing batching."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import shutil

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
    NTUPLE_NAME,
    SPECTRUM_HIST_NAME,
    final_output_path,
    output_stem,
    temp_work_dir,
)

DEFAULT_SEED = 908136382


def apply_verbosity(run_manager: G4RunManager, verbose: int) -> None:
    run_manager.SetVerboseLevel(verbose)
    G4HadronicParameters.Instance().SetVerboseLevel(verbose)
    G4EmParameters.Instance().SetVerbose(verbose)
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(f"/run/verbose {verbose}")


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
    G4Random.setTheSeed(int(seed))

    mats = materials.build_all_materials(spec)
    det = detector.DetectorConstruction(
        spec, mats, check_overlaps=verbose > 0)

    # Silence pybind11's static-initialization chatter on first import.
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            run_manager = G4RunManagerFactory.CreateRunManager(
                G4RunManagerType.Serial)

    run_manager.SetUserInitialization(det)
    run_manager.SetUserInitialization(physics.PhysicsList())
    run_manager.SetUserInitialization(
        actions.ActionInitialization(
            spec, det, output_stem, event_offset, verbose)
    )
    apply_verbosity(run_manager, verbose)
    run_manager.SetPrintProgress(5000)
    run_manager.Initialize()
    physics.configure_radioactive_decay(spec)
    physics.configure_gps(spec, det)
    return run_manager


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


def _split_events(n_events: int, n_parts: int) -> list[int]:
    base, remainder = divmod(n_events, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _validate_merged_output(output_path: str, expected_entries: int) -> None:
    """Check the hadd output and remove it when it is incomplete.

    A failed or aborted merge can leave a truncated-but-readable file at
    the production path; later runs would skip it as an existing output.
    Require the ntuple and spectrum histogram to be present and the ntuple
    entry count to match the sum of the worker entries.
    """
    import uproot

    reason = None
    try:
        with uproot.open(output_path) as f:
            if NTUPLE_NAME not in f:
                reason = "missing ntuple"
            elif SPECTRUM_HIST_NAME not in f:
                reason = "missing spectrum histogram"
            else:
                n_entries = int(f[NTUPLE_NAME].num_entries)
                if n_entries != expected_entries:
                    reason = (f"{n_entries} ntuple entries, expected "
                              f"{expected_entries}")
    except Exception as exc:
        reason = f"unreadable output: {exc}"
    if reason is not None:
        _remove_file(output_path)
        raise RuntimeError(
            f"hadd output {output_path!r} failed validation ({reason})")


def merge_worker_outputs(output_path: str, input_paths: list[str], *,
                         hadd_exe: str | None = None) -> int:
    """Merge worker ROOT files into the final simulation output via hadd.

    Delegates to :func:`kc761util.hadd.merge_root_files`, which merges the
    worker ntuples entry-by-entry and sums the spectrum histograms together
    with their ``sumw2`` buffers, so the merged file carries the Monte Carlo
    statistical errors used by the calibration fit.  Before merging, the
    worker spectrum histogram binnings are validated: hadd silently adds
    differently binned histograms bin-by-bin, which would corrupt the merged
    spectrum, so a mismatch is a hard error here.  After merging, the output
    is validated (ntuple and histogram present, entry count matches the
    workers) and any partial output from a failed merge is removed.
    """
    import numpy as np
    import uproot

    from kc761util.hadd import merge_root_files
    from kc761util.spectrum import load_spectrum

    if not input_paths:
        raise ValueError("merge_worker_outputs: no worker files to merge")

    ref_edges = None
    expected_entries = 0
    for path in input_paths:
        with uproot.open(path) as src:
            if NTUPLE_NAME not in src:
                raise RuntimeError(
                    f"ntuple {NTUPLE_NAME!r} missing in worker file {path!r}")
            expected_entries += int(src[NTUPLE_NAME].num_entries)
            try:
                spectrum = load_spectrum(src)
            except KeyError as exc:
                raise RuntimeError(
                    f"spectrum histogram {SPECTRUM_HIST_NAME!r} missing in "
                    f"worker file {path!r}") from exc
        edges = spectrum.edges
        if ref_edges is None:
            ref_edges = edges
        elif not np.array_equal(edges, ref_edges):
            raise RuntimeError(
                f"spectrum histogram bin edges differ between worker "
                f"files ({path!r} disagrees with earlier workers)")

    rc = merge_root_files(output_path, input_paths, hadd_exe=hadd_exe)
    if rc != 0:
        _remove_file(output_path)
        return rc
    _validate_merged_output(output_path, expected_entries)
    return 0


def run_batch(
    source_key: str,
    output_path: str,
    n_events: int,
    threads: int,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
    hadd_exe: str | None = None,
) -> None:
    """Run a simulation on ``threads`` workers and merge their outputs."""
    stem = output_stem(output_path)
    final_path = final_output_path(output_path)

    if threads <= 1:
        run_simulation(source_key, stem, n_events, seed, 0, verbose)
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
                        run_simulation,
                        (
                            source_key,
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
                hadd_exe=hadd_exe)
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
