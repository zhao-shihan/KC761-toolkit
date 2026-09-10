"""Matrix-mode simulation-file output: the Monte Carlo primary-to-deposition step.

Pipeline glue for the kc761sim matrix modes: reads the merged G ROOT file
(the hadd output of the worker histograms), validates it against the
input calibration file's deposition-energy binning (the calibration file
is a binning reference only -- no composition happens here), and writes
the final G ROOT file through :file:`kc761sim/sim2root.cxx` (the
temporary-binary export convention of kc761calib/kc761unfold):

    gamma(E_gamma) -> crystal deposition (physics, G).

The primary-to-channel composition R = C @ G happens at unfold time
(:mod:`kc761unfold`), which receives both the calibration file and this
simulation file.  It carries the Monte Carlo counts with their per-bin
variance (the Gaussian approximation ``var ~ counts``), the per-primary
-column detection efficiency ``1 - zero-deposition fraction`` and the
run metadata.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import uproot

from kc761util.binexport import ExportWriter
from kc761util.calibfile import MAX_CHANNELS, load_calib_file
from kc761util.rootcxxfrontend import format_macro_cmd, run_macro

from .paths import PRIMARY_TO_DEPOSITION_HIST_NAME, ZERO_DEPOSITION_HIST_NAME
from .sources import MatrixSource, mode_metadata

_MAGIC = b"kc761sim-sim-export-v1\n"


def _load_histograms(path: str) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray]:
    """Read the merged primary-to-deposition counts, zero counts and the three edge arrays.

    uproot returns ``values()[deposition_bin, primary_bin]`` -- exactly the
    layout the output works on (the G x axis is the matrix output side,
    the y axis the input side).  Returns ``(counts, zero_counts,
    x_edges, y_edges, z_edges)``: the G x-axis and y-axis edge arrays
    (identical by construction) and the zero-deposition axis edges.
    """
    with uproot.open(path) as f:
        try:
            g_hist = f[PRIMARY_TO_DEPOSITION_HIST_NAME]
            zero_hist = f[ZERO_DEPOSITION_HIST_NAME]
        except KeyError as exc:
            raise RuntimeError(
                f"merged file {path!r} is missing the worker histograms "
                f"({exc.args[0]}); is it a kc761sim matrix-mode output?") \
                from exc
        # Validate the bin counts BEFORE materializing the dense contents
        # (a corrupt file can declare huge axes whose sparse content
        # compresses to a tiny file; the same cap the calibration reader
        # enforces).
        n_x = len(g_hist.axis(0).edges()) - 1
        n_y = len(g_hist.axis(1).edges()) - 1
        if n_x != n_y or n_x < 1 or n_x > MAX_CHANNELS:
            raise ValueError(
                f"G histogram bin counts {n_x} x {n_y} are inconsistent "
                f"or beyond the supported maximum of {MAX_CHANNELS} in "
                f"{path}")
        if len(zero_hist.axis(0).edges()) - 1 != n_x:
            raise ValueError(
                f"zero-deposition histogram bin count does not match the "
                f"G histogram ({n_x}) in {path}")
        counts = np.asarray(
            g_hist.values(), dtype=float)  # [deposition, primary]
        zero_counts = np.asarray(zero_hist.values(), dtype=float)
        x_edges = np.asarray(g_hist.axis(0).edges(), dtype=float)
        y_edges = np.asarray(g_hist.axis(1).edges(), dtype=float)
        z_edges = np.asarray(zero_hist.axis(0).edges(), dtype=float)
    return counts, zero_counts, x_edges, y_edges, z_edges


def _validate_inputs(calib, counts, zero_counts, x_edges, y_edges,
                     z_edges, source, n_events) -> np.ndarray:
    """Fail fast on every binning/accounting inconsistency before writing.

    Returns the per-column primary-event totals (zero-deposition events
    included).
    """
    n = calib.n_channels
    edges = calib.energy_edges
    if not np.array_equal(np.asarray(source.axis.edges), edges):
        raise ValueError(
            "the source primary axis does not match the calibration "
            "deposition-energy edges of the input file")
    if counts.shape != (n, n):
        raise ValueError(
            f"G histogram shape {counts.shape} does not match the "
            f"calibration matrix {(n, n)}")
    if not np.array_equal(x_edges, edges) or not np.array_equal(y_edges, edges):
        raise ValueError(
            "G histogram binning does not match the calibration "
            "deposition-energy edges (both axes must reuse them exactly)")
    if zero_counts.shape != (n,) or not np.array_equal(z_edges, edges):
        raise ValueError(
            "zero-deposition histogram binning does not match the "
            "calibration deposition-energy edges")
    if (counts < 0).any() or (zero_counts < 0).any():
        raise ValueError("G or zero-deposition counts contain negative values")

    totals = counts.sum(axis=0) + zero_counts
    if int(totals.sum()) != n_events:
        raise ValueError(
            f"total scored events {int(totals.sum())} do not match the "
            f"requested n_events {n_events}")
    axis = source.axis
    expected = n_events // axis.n_active
    active = np.array(axis.active, dtype=int)
    if np.any(np.abs(totals[active] - expected) > 1.5):
        raise ValueError(
            "per-column event counts deviate from the fixed-per-column "
            f"scheme (expected {expected} +- 1 per active column)")
    if (totals[np.setdiff1d(np.arange(n), active)] != 0).any():
        raise ValueError(
            "skipped (negative-energy) primary columns received events")
    return totals


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sim_export(calib, counts, totals, source,
                         n_events, seed, calib_path: str) -> str:
    """Serialize everything sim2root.cxx needs into a temporary file.

    Returns the temporary file path; the ROOT writer deletes it after a
    successful conversion, and callers keep it on failure for inspection
    (the kc761calib convention).
    """
    n = calib.n_channels
    mode, mode_name, geometry_name, geometry_param = mode_metadata(source)
    edges = calib.energy_edges

    # Detection efficiency per primary column: the complement of the
    # zero-deposition fraction.
    efficiency = np.divide(counts.sum(axis=0), totals,
                           out=np.zeros(n), where=totals > 0.0)

    writer = ExportWriter("kc761sim-sim-export-", _MAGIC)
    with writer:
        writer.put_text(mode_name)
        writer.put_text(geometry_name)
        writer.put_text(source.angular)
        writer.put_text(os.path.abspath(calib_path))
        writer.put_text(_sha256(calib_path))
        writer.put_i64(n)
        writer.put_i64(mode)
        writer.put_f64(edges)
        writer.put_f64(geometry_param)
        writer.put_f64([n_events, seed])
        # The block is [deposition, primary] row-major; sim2root.cxx
        # fills the TH2D from it directly.  Block 2 repeats the counts
        # as the unit-weight sumw2 buffer (var ~ counts).
        writer.put_block(1, counts.ravel())
        writer.put_block(2, counts.ravel())
        writer.put_block(3, efficiency)
    return writer.path


def write_sim_file(
    calib_path: str,
    merged_g_path: str,
    output_path: str,
    source: MatrixSource,
    n_events: int,
    seed: int,
    root_exe: str | None = None,
    calib=None,
) -> int:
    """Write the simulation ROOT file from the merged worker histograms.

    Prints progress like the other frontends; on a ROOT failure the
    temporary export file is kept and the manual re-run command printed.
    ``calib`` optionally carries an already-loaded calibration snapshot
    (the CLI pre-validates the input and passes it to avoid decompressing
    the dense matrix twice); when omitted, the file is loaded here so the
    function stays self-contained for re-running after a crash.
    """
    if calib is None:
        calib = load_calib_file(calib_path)
    counts, zero_counts, x_edges, y_edges, z_edges = _load_histograms(
        merged_g_path)
    totals = _validate_inputs(calib, counts, zero_counts, x_edges,
                              y_edges, z_edges, source, n_events)

    print(f"[sim] writing the {calib.n_channels} x {calib.n_channels} "
          f"primary-to-deposition matrix (composition happens at unfold "
          f"time)")

    # Physical sanity of the G contents before writing anything.
    active = np.asarray(source.axis.active, dtype=int)
    efficiency = np.divide(counts.sum(axis=0), totals,
                           out=np.zeros(calib.n_channels),
                           where=totals > 0.0)
    if (efficiency < 0.0).any() or (efficiency > 1.0).any():
        raise RuntimeError(
            "detection efficiency outside [0, 1] (internal inconsistency)")
    print(f"[sim] detection efficiency: min "
          f"{efficiency[active].min():.4g} max "
          f"{efficiency[active].max():.4g} "
          f"(over {active.size} active primary columns)")

    export_file = write_sim_export(
        calib, counts, totals, source,
        n_events, seed, calib_path)
    rc = run_macro("kc761sim/sim2root.cxx",
                   [export_file, str(output_path)],
                   root_exe=root_exe, echo_prefix="sim2root")
    if rc != 0:
        cmd = format_macro_cmd("kc761sim/sim2root.cxx",
                               [export_file, str(output_path)],
                               root_exe=root_exe)
        re_run = (f"\n[sim]   manual re-run: {' '.join(cmd)}"
                  if cmd is not None else "")
        print(f"[sim] error: ROOT export failed (exit code {rc}); the "
              f"temporary export file was kept for inspection:"
              f"\n[sim]   {export_file}{re_run}",
              file=sys.stderr)
    return rc
