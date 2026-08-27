"""Run orchestration: single-process simulations and multiprocessing batching."""

from __future__ import annotations

import multiprocessing
import contextlib
import os
import shutil

from geant4_pybind import (
    G4EmParameters,
    G4HadronicParameters,
    G4Random,
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
from .paths import final_output_path, output_stem, temp_work_dir

DEFAULT_SEED = 908136382


def apply_verbosity(run_manager, verbose: int) -> None:
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
    G4Random.setTheSeed(int(seed))

    spec = config.SOURCES[source_key]
    mats = materials.build_all_materials()
    det = detector.DetectorConstruction(spec, mats, check_overlaps=verbose > 0)

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
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
    run_manager.BeamOn(n_events)
    return n_events


def _split_events(n_events: int, n_parts: int) -> list[int]:
    base, remainder = divmod(n_events, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def merge_root_files(output_path: str, input_paths: list[str]) -> None:
    import numpy as np
    import uproot

    tree_name = actions.NTUPLE_NAME
    hist_name = actions.SPECTRUM_HIST_NAME

    if not input_paths:
        raise ValueError("merge_root_files: no worker files to merge")

    hist_values = None
    hist_edges = None

    with uproot.recreate(output_path) as f:
        tree = f.mktree(
            tree_name,
            {
                "event_id": np.int32,
                "edep": np.float32,
                "time": np.float64,
            },
            title="kc761_data - per-pulse energy deposition",
        )
        for path in input_paths:
            with uproot.open(path) as src:
                src_tree = src[tree_name]
                for batch in src_tree.iterate(
                    ["event_id", "edep", "time"],
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
                hist_values = (
                    values if hist_values is None else hist_values + values
                )
                hist_edges = edges
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
        for i, (chunk, offset) in enumerate(zip(chunks, offsets)):
            if chunk <= 0:
                continue
            worker_stems.append(
                os.path.join(work_dir, f"{base}-kc761sim_w{i}_wip"))
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
    finally:
        pool.terminate()

    worker_paths = [s + ".root" for s in worker_stems]
    try:
        merge_root_files(final_path, worker_paths)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
