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
    NTUPLE_COLUMNS,
    NTUPLE_NAME,
    SPECTRUM_HIST_NAME,
    final_output_path,
    ntuple_title,
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
    run_manager.SetPrintProgress(1000)
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


def merge_root_files(output_path: str, input_paths: list[str]) -> None:
    """Merge worker ROOT files into one ntuple plus summed histogram."""
    import numpy as np
    import uproot

    tree_name = NTUPLE_NAME
    hist_name = SPECTRUM_HIST_NAME

    if not input_paths:
        raise ValueError("merge_root_files: no worker files to merge")

    columns = list(NTUPLE_COLUMNS)
    hist_values = None
    hist_edges = None

    with uproot.recreate(output_path) as f:
        tree = f.mktree(
            tree_name, dict(NTUPLE_COLUMNS), title=ntuple_title()
        )
        for path in input_paths:
            with uproot.open(path) as src:
                src_tree = src[tree_name]
                for batch in src_tree.iterate(
                    columns,
                    library="np",
                    step_size="10 MB",
                ):
                    tree.extend(batch)
                if hist_name not in src:
                    raise RuntimeError(
                        f"spectrum histogram {hist_name!r} missing in worker "
                        f"file {path!r}"
                    )
                values, edges = src[hist_name].to_numpy()
            if hist_values is None:
                hist_values, hist_edges = values, edges
            elif not np.array_equal(edges, hist_edges):
                raise RuntimeError(
                    f"spectrum histogram bin edges differ between worker "
                    f"files ({path!r} disagrees with earlier workers)"
                )
            else:
                hist_values = hist_values + values
        if hist_values is not None:
            f[hist_name] = (hist_values, hist_edges)


def run_batch(
    source_key: str,
    output_path: str,
    n_events: int,
    threads: int,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
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
            merge_root_files(final_path, [s + ".root" for s in worker_stems])
        finally:
            pool.terminate()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
