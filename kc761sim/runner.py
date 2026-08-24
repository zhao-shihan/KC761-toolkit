"""Run orchestration: single-process simulations and multiprocessing batching.

Because of the Python GIL, geant4_pybind cannot benefit from Geant4's native
multithreading (an ``G4MTRunManager`` built with Python user actions deadlocks,
see the geant4_pybind README).  Parallelism is therefore achieved with Python
``multiprocessing``: each worker runs its own serial Geant4 instance over a
chunk of the requested events, writes its own ROOT file, and the master merges
the chunks with ``uproot`` into the final output file.
"""

from __future__ import annotations

import multiprocessing
import os

from geant4_pybind import (
    G4EmParameters,
    G4HadronicParameters,
    G4Random,
    G4RunManagerFactory,
    G4RunManagerType,
    G4UImanager,
)

from . import actions, config, detector, materials, physics

#: default random seed used when --seed is not given (worker i uses seed + i + 1)
DEFAULT_SEED = 123456789


def _output_stem(path: str) -> str:
    """Strip a trailing ``.root`` (case-insensitive) for G4Analysis, which
    appends the file-type extension itself."""
    lower = path.lower()
    if lower.endswith(".root"):
        return path[:-5]
    return path


def _final_output_path(path: str) -> str:
    return _output_stem(path) + ".root"


def apply_verbosity(run_manager, verbose: int) -> None:
    """Set the Geant4 verbosity level (0 = only banner and run progress).

    Must be called after the run manager is created and before
    ``G4RunManager::Initialize`` so that the physics-construction messages
    (hadronic and EM parameter dumps, HP data loading, ...) are suppressed
    too.  ``/tracking/verbose`` is intentionally NOT set: it stays at its
    default (0) so that per-track output never floods the run.
    """
    run_manager.SetVerboseLevel(verbose)
    G4HadronicParameters.Instance().SetVerboseLevel(verbose)
    G4EmParameters.Instance().SetVerbose(verbose)
    ui = G4UImanager.GetUIpointer()
    ui.ApplyCommand(f"/run/verbose {verbose}")


def run_simulation(
    source_key: str,
    output_stem: str,
    n_events: int,
    seed: int,
    event_offset: int = 0,
    verbose: int = 0,
) -> int:
    """Run one serial Geant4 simulation in the current process.

    Writes the ROOT ntuple to ``<output_stem>.root``.  Intended to be called
    from a worker process (or directly when ``--threads 1``).
    """
    G4Random.setTheSeed(int(seed))

    spec = config.SOURCES[source_key]
    mats = materials.build_all_materials()
    det = detector.DetectorConstruction(spec, mats, check_overlaps=verbose > 0)

    run_manager = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    run_manager.SetUserInitialization(det)
    run_manager.SetUserInitialization(physics.PhysicsList())
    run_manager.SetUserInitialization(
        actions.ActionInitialization(
            spec, det, output_stem, event_offset, verbose)
    )
    apply_verbosity(run_manager, verbose)
    run_manager.SetPrintProgress(max(1, n_events // 10))
    run_manager.Initialize()
    physics.configure_radioactive_decay(spec)
    physics.configure_gps(spec, det)
    run_manager.BeamOn(n_events)
    return n_events


def _split_events(n_events: int, n_parts: int) -> list[int]:
    """Distribute ``n_events`` over ``n_parts`` workers as evenly as possible."""
    base, remainder = divmod(n_events, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def merge_root_files(output_path: str, input_paths: list[str]) -> None:
    """Merge the per-worker ROOT ntuples (and the spectrum histogram) into a
    single output file, preserving the float32 energy-deposition column."""
    import numpy as np
    import uproot

    tree_name = actions.NTUPLE_NAME
    hist_name = actions.SPECTRUM_HIST_NAME

    event_ids: list[np.ndarray] = []
    edeps: list[np.ndarray] = []
    times: list[np.ndarray] = []
    hist_values = None
    hist_edges = None

    for path in input_paths:
        with uproot.open(path) as f:
            tree = f[tree_name]
            arrays = tree.arrays(library="np")
            event_ids.append(arrays["event_id"])
            edeps.append(arrays["edep"])
            times.append(arrays["time"])
            if hist_name in f:
                values, edges = f[hist_name].to_numpy()
                hist_values = (
                    values if hist_values is None else hist_values + values
                )
                hist_edges = edges

    with uproot.recreate(output_path) as f:
        # uproot >= 5.7 writes RNTuples by default; use mktree so the merged
        # output stays a classic TTree (readable by any ROOT version).
        tree = f.mktree(
            tree_name,
            {
                "event_id": np.int32,
                "edep": np.float32,
                "time": np.float64,
            },
            title="kc761_data - per-event energy deposition",
        )
        tree.extend(
            {
                "event_id": np.concatenate(event_ids),
                "edep": np.concatenate(edeps),
                "time": np.concatenate(times),
            }
        )
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
    """Run ``n_events`` events for ``source_key`` using ``threads`` processes
    and write the merged result to ``output_path``."""
    stem = _output_stem(output_path)
    final_path = _final_output_path(output_path)

    if threads <= 1:
        run_simulation(source_key, stem, n_events, seed, 0, verbose)
        return

    chunks = _split_events(n_events, threads)
    offsets = []
    running = 0
    for chunk in chunks:
        offsets.append(running)
        running += chunk

    worker_stems: list[str] = []
    tasks: list[multiprocessing.pool.AsyncResult] = []
    pool = multiprocessing.Pool(threads)
    try:
        for i, (chunk, offset) in enumerate(zip(chunks, offsets)):
            if chunk <= 0:
                continue
            worker_stems.append(f"{stem}_w{i}")
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
            task.get()  # re-raise worker exceptions in the master
    finally:
        pool.terminate()

    worker_paths = [s + ".root" for s in worker_stems]
    try:
        merge_root_files(final_path, worker_paths)
    finally:
        for path in worker_paths:
            if os.path.exists(path):
                os.remove(path)
