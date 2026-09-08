"""Hybrid regularized unfolding: banded active-set QP and Hessian algebra.

The objective

    chi2(mu) + alpha ||D mu||^2,   chi2 = sum_j (y_j - (R mu)_j)^2 / sigma_j^2

under ``mu >= 0`` is a convex quadratic program.  The Hessian
``H = 2 R^T W R + 2 a D^T D`` is banded (for calibration responses the
bandwidth is ~2 x the Gaussian kernel support; composite responses are
wider but use the same machinery), so the bound-constrained optimum is
found with a Lawson-Hanson style active-set loop on LAPACK banded
Cholesky factorizations (O(n b^2), BLAS-parallel): bins with negative
reduced gradient enter the free set, and the unrestricted optimum of
the free set is approached along a feasible segment that stops at the
first bin hitting zero (which is dropped), terminating at the exact KKT
point ``mu >= 0, g >= 0, mu * g = 0``.  The same banded machinery serves
the analytic error propagation (:mod:`kc761unfold.errors`), keeping the
solver and the covariance computation on one source of truth.
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded

JITTER_FRAC = 1e-10  # per-entry relative jitter, fallback only (non-PD Hessian)


# --- banded storage helpers ---


def csr_to_upper_banded(h: sparse.csr_matrix) -> tuple[np.ndarray, int]:
    """Convert a symmetric CSR matrix to LAPACK upper-banded storage.

    Returns ``(ab, u)`` with ``ab[u + i - j, j] = h[i, j]`` for the upper
    triangle (i <= j); ``u`` is the number of super-diagonals.
    """
    n = h.shape[0]
    coo = h.tocoo()
    i = coo.row
    j = coo.col
    sel = j >= i
    u = int((j[sel] - i[sel]).max())
    ab = np.zeros((u + 1, n), dtype=float)
    # direct assignment: every call site passes a canonical CSR (no
    # duplicate (i, j) pairs), so the add.at duplicate-summation
    # semantics are not needed here.
    ab[u + i[sel] - j[sel], j[sel]] = coo.data[sel]
    return ab, u


@numba.njit(cache=True)
def _embed_equilibrate(ab: np.ndarray, u: int, free: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Embed identity rows/cols for non-free bins and equilibrate.

    Fused compiled version of the two O(u) Python loops that previously
    dominated the factorization cost: band entries ``(i, j)`` of the
    upper triangle with a non-free row or column are zeroed, the
    diagonal is set to 1 on non-free bins, and the block is
    symmetrically scaled by ``d = 1/sqrt(|diag|)`` (band entry
    ``(i, j)`` scaled by ``d_i d_j``; padding entries below the band are
    already zero).  Returns the prepared banded block and ``d``.
    """
    n = ab.shape[1]
    abw = ab.copy()
    for d in range(1, u + 1):
        for j in range(n):
            i = j - d
            if i < 0:
                continue
            if (not free[j]) or (not free[i]):
                abw[u - d, j] = 0.0
    d_scale = np.empty(n, dtype=np.float64)
    for j in range(n):
        if not free[j]:
            abw[u, j] = 1.0
        d_scale[j] = 1.0 / np.sqrt(max(np.abs(abw[u, j]), 1e-30))
    for k in range(u + 1):
        shift = u - k
        for j in range(n):
            i = j - shift
            if i >= 0:
                abw[k, j] *= d_scale[i] * d_scale[j]
    return abw, d_scale


def embed_and_factor(ab: np.ndarray, u: int, free: np.ndarray
                     ) -> tuple[np.ndarray, bool, np.ndarray]:
    """Embed identity rows/cols for non-free bins, equilibrate, factorize.

    Returns the Cholesky factor, the lower flag and the diagonal
    equilibration vector ``d``: the embedded block ``H_emb`` is
    symmetrically scaled to ``diag(d) H_emb diag(d)`` with
    ``d = 1/sqrt(diag)`` before the factorization and
    :func:`solve_embedded` undoes the scaling on the solution.  The
    Hessian diagonal spans many orders of magnitude
    (w = 1/sigma^2 with sigma ~ 0.1 y, giving cond(H) ~ 1e10 for
    full-range spectra), which would limit the plain banded Cholesky to
    ~kappa * eps relative accuracy; diagonal equilibration preserves the
    band structure and cuts the condition number to ~1e6, restoring full
    precision (this is what keeps the active-set decisions exact at the
    mu = 0 boundary).  No jitter is added on the normal path; only a
    non-positive-definite embedded block (degenerate data) falls back to
    a per-entry relative jitter.
    """
    abw, d_scale = _embed_equilibrate(ab, u, free)
    try:
        c = cholesky_banded(abw, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        abw[u] *= 1.0 + JITTER_FRAC
        c = cholesky_banded(abw, lower=False, check_finite=False)
    if not np.isfinite(c).all():
        abw[u] *= 1.0 + JITTER_FRAC
        c = cholesky_banded(abw, lower=False, check_finite=False)
    return c, False, d_scale


def solve_embedded(ab: np.ndarray, u: int, free: np.ndarray, rhs: np.ndarray,
                   factor: tuple[np.ndarray, bool, np.ndarray] | None = None
                   ) -> np.ndarray:
    """Solve the embedded (equilibrated) banded system.

    ``rhs`` may be one vector or a column stack of right sides; the
    equilibration applied by :func:`embed_and_factor` is undone on the
    solution.
    """
    if factor is None:
        factor = embed_and_factor(ab, u, free)
    c, lower, d_scale = factor
    if rhs.ndim == 1:
        b = (np.where(free, rhs, 0.0)) * d_scale
    else:
        b = (rhs * free[:, None]) * d_scale[:, None]
    x = cho_solve_banded((c, lower), b)
    return x * d_scale[:, None] if x.ndim == 2 else x * d_scale


# --- problem and solve ---


class UnfoldProblem:
    """One unfolding problem: the banded QP and its Hessian machinery."""

    def __init__(self, r: sparse.csr_matrix, y: np.ndarray, w: np.ndarray,
                 d_op: sparse.csr_matrix, alpha: float):
        self.r = r
        self.y = y
        self.w = w
        self.d_op = d_op
        self.alpha = alpha
        self.n = len(y)
        # Full CSR Hessian H = 2 R^T W R + 2 a D^T D and the
        # normal-equation right side b = 2 R^T W y (both mu-independent).
        self._h = (2.0 * (r.T @ sparse.diags(w) @ r)
                   + 2.0 * alpha * (d_op.T @ d_op)).tocsr()
        self._b = 2.0 * (self.r.T @ (self.w * self.y))

    # --- objective ---

    def _chi2(self, mu: np.ndarray) -> float:
        r_ = self.y - self.r @ mu
        return float(r_ @ (self.w * r_))

    def hessian_banded(self) -> tuple[np.ndarray, int]:
        """Full Hessian 2 R^T W R + 2 a D^T D, banded (mu-independent)."""
        return csr_to_upper_banded(self._h)

    def solve(self) -> np.ndarray:
        """Minimize the unfolding objective under ``mu >= 0``.

        A Lawson-Hanson style active-set loop on the banded Cholesky,
        block version: every bin whose reduced gradient ``g = H mu - b``
        is negative enters the free set (the unrestricted optimum over
        the enlarged set strictly decreases the objective); then the
        inner loop walks from the current *feasible* ``mu`` along the
        segment ``mu + t (x_P - mu)`` to the optimum ``x_P`` of the free
        set, stopping at the first bin that hits zero (``t < 1``), which
        is dropped and the set re-solved.  The iterate stays feasible and
        the objective strictly decreases at every solve, so the loop
        converges to the exact KKT point ``mu >= 0, g >= 0, mu * g = 0``
        (a stall guard catches the degenerate ``t = 0`` corner).  Fills
        ``mu``, ``free``, ``chi2``, ``pen_cost`` and ``n_iter``; returns
        ``mu``.
        """
        ab, u = self.hessian_banded()
        n = self.n
        max_iters = 2 * n + 100
        mu = np.zeros(n)
        free = np.zeros(n, dtype=bool)
        prev_state = None
        for nit in range(1, max_iters):
            g = self._h @ mu - self._b
            scale = max(1.0, float(np.abs(g).max()))
            enter = (~free) & (g < -1e-10 * scale)
            if not enter.any():
                break
            free = free | enter
            while True:
                factor = embed_and_factor(ab, u, free)
                xp = solve_embedded(ab, u, free, self._b, factor=factor)
                viol = free & (xp < 0.0)
                if not viol.any():
                    mu = xp
                    break
                # walk mu -> xp until the first dropped bin hits zero
                t = float(np.min(mu[viol] / (mu[viol] - xp[viol])))
                mu = mu + t * (xp - mu)
                hit = viol & (mu <= 1e-12 * max(1.0, float(np.max(mu))))
                free = free & ~hit
            # Cycle guard: an outer iteration that changed neither mu nor
            # the free set (the t = 0 degenerate walk) would repeat
            # identically forever; report the stalled point instead.
            state = (free.tobytes(), mu.tobytes())
            if state == prev_state:
                print("[unfold] warning: active-set QP stalled without "
                      "progress; the solution may be degenerate")
                break
            prev_state = state
        else:
            print(f"[unfold] warning: active-set QP did not converge in "
                  f"{max_iters - 1} iterations")
        self.n_iter = nit
        clip_tol = max(1e-6, 1e-9 * float(np.max(mu)))
        self.mu = np.where(mu < clip_tol, 0.0, mu)
        # the covariance machinery (kc761unfold.errors) treats bins at the
        # bound as fixed directions, exactly like the stored solution.
        self.free = self.mu > 0.0
        self.chi2 = self._chi2(self.mu)
        d = self.d_op @ self.mu
        self.pen_cost = float(d @ d)
        return self.mu
