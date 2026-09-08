"""Console summary of an UnfoldResult."""

from __future__ import annotations

import numpy as np

from .types import UnfoldResult


def print_summary(result: UnfoldResult, data_path: str,
                  calib_path: str) -> None:
    """Print the one-shot report lines to stdout."""
    s = result.settings
    print(f"[unfold] data = {data_path}")
    print(f"[unfold] calibration = {calib_path}")
    if result.calib_only:
        print("[unfold] calibration-only")
    else:
        print(f"[unfold] hybrid: alpha={s.alpha:g}, k={s.k}, "
              f"snip_iter={s.snip_iter}, mask_z0={s.mask_z0:g}, "
              f"mask_floor={s.mask_floor:g}, syst={s.syst_frac:g}")
    print(f"[unfold] energy {result.energy_edges[0]:.2f} - "
          f"{result.energy_edges[-1]:.2f} keV "
          f"(channels {result.channel_low}-{result.channel_high}, "
          f"{result.n_bins} bins)")
    if result.calib_only:
        print(f"[unfold] uncertainties: stored stat max "
              f"{np.max(result.sigma_stat):.4g}, calibration vertical max "
              f"{np.max(result.sigma_calib):.4g}")
        return
    print(f"[unfold] chi2 = {result.chi2:.2f}, ndof = {result.ndof}, "
          f"chi2/ndof = {result.chi2 / result.ndof:.2f}, "
          f"penalty = {result.pen_cost:.4g} "
          f"({result.n_iter} iterations)")
    total_in = float(np.sum(result.data_counts))
    total_out = float(np.sum(result.counts))
    print(f"[unfold] counts: data {total_in:.1f} -> unfolded "
          f"{total_out:.1f} (ratio {total_out / total_in:.4f})")
    print(f"[unfold] uncertainties: stat max "
          f"{np.max(result.sigma_stat):.4g}, syst max "
          f"{np.max(result.sigma_syst):.4g}, total max "
          f"{np.max(result.sigma_total):.4g}")
