"""SWR unfolding solver: banded linear algebra and optimization.

The response matrix, the penalty operators and the Hessians are all
banded (nonzero entries confined to a diagonal band of width ~ 2 x the
Gaussian kernel support, ~420 channels out of 2048).  The module builds
the Hessian in LAPACK upper-banded storage and solves with the banded
Cholesky (O(n b^2) instead of O(n^3)), which makes every Newton step,
the quadratic warm-start and each covariance solve in kc761unfold.errors
subsecond.

The Huber objective is minimized by a damped Newton method with an
active-set treatment of the positivity constraint: the Hessian
2 R^T W R + a D^T diag(rho''(d)) D is positive semi-definite on the
free set, the embedded banded factorization gives a descent direction,
and a backtracking line search with clipping handles the bound mu >= 0.
L-BFGS-B over a softplus parameterization converges far more slowly on
this problem (the clipped-bin directions are exponentially flat in the
link-function space).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded

from .penalty import huber_grad, huber_hess, huber_value

JITTER_FRAC = 1e-10  # relative diagonal jitter for the banded Cholesky


# --------------------------------------------------------------------------
# banded storage helpers


def csr_to_upper_banded(h: sparse.csr_matrix, n: int
                        ) -> tuple[np.ndarray, int]:
    """Convert a symmetric CSR matrix to LAPACK upper-banded storage.

    Returns ``(ab, u)`` with ``ab[u + i - j, j] = h[i, j]`` for the upper
    triangle (i <= j); ``u`` is the number of super-diagonals.
    """
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


def embed_and_factor(ab: np.ndarray, u: int, free: np.ndarray
                     ) -> tuple[np.ndarray, bool]:
    """Embed identity rows/cols for non-free bins, jitter, factorize.

    Returns the Cholesky factor and the lower flag for cho_solve_banded.
    """
    n = ab.shape[1]
    abw = ab.copy()
    nf = np.where(~free)[0]
    if nf.size:
        for d in range(1, u + 1):
            abw[u - d, nf] = 0.0  # entries (k - d, k)
            cols = nf + d
            ok = cols < n
            if ok.any():
                abw[u - d, cols[ok]] = 0.0  # entries (k, k + d)
    abw[u] = np.where(free, abw[u], 1.0)
    abw[u] += JITTER_FRAC * max(1.0, float(np.abs(abw[u]).max()))
    c = cholesky_banded(abw, lower=False, check_finite=False)
    return c, False


def solve_embedded(ab: np.ndarray, u: int, free: np.ndarray, rhs: np.ndarray,
                   factor: tuple[np.ndarray, bool] | None = None
                   ) -> np.ndarray:
    """Solve the embedded banded system against one or several right sides."""
    if factor is None:
        factor = embed_and_factor(ab, u, free)
    b = np.where(free, rhs, 0.0) if rhs.ndim == 1 else rhs * free[:, None]
    return cho_solve_banded(factor, b)


# --------------------------------------------------------------------------
# problem and optimization


def swr_gradient(r: sparse.csr_matrix, y: np.ndarray, w: np.ndarray,
                 d_op: sparse.csr_matrix, alpha: float, delta: float,
                 mu: np.ndarray) -> np.ndarray:
    """dL/dmu at fixed mu for the SWR objective (Huber clip included).

    The single source of the objective gradient: the solver's Newton
    loop evaluates it on the nominal operators, and the systematic-
    covariance finite differences in kc761unfold.errors evaluate it on
    the perturbed ones.
    """
    r_ = y - r @ mu
    d = d_op @ mu
    return (-2.0 * (r.T @ (w * r_))
            + alpha * (d_op.T @ huber_grad(d, delta)))


class SWRProblem:
    """One unfolding problem: SWR objective over the channel subrange."""

    def __init__(self, r: sparse.csr_matrix, y: np.ndarray, w: np.ndarray,
                 d_op: sparse.csr_matrix, alpha: float, delta: float):
        self.r = r
        self.y = y
        self.w = w
        self.d_op = d_op
        self.alpha = alpha
        self.delta = delta
        self.n = len(y)
        # constant data part of the Hessian (does not depend on mu)
        self._rtwr = (2.0 * (r.T @ sparse.diags(w) @ r)).tocsr()

    # -- objective --------------------------------------------------------

    def _chi2(self, mu: np.ndarray) -> float:
        r_ = self.y - self.r @ mu
        return float(r_ @ (self.w * r_))

    def grad_mu(self, mu: np.ndarray) -> np.ndarray:
        """dL/dmu at fixed mu on the nominal operators."""
        return swr_gradient(self.r, self.y, self.w, self.d_op, self.alpha,
                            self.delta, mu)

    def _loss(self, mu: np.ndarray) -> float:
        return self._chi2(mu) + self.alpha * float(
            huber_value(self.d_op @ mu, self.delta).sum())

    def hessian_banded(self, mu: np.ndarray
                       ) -> tuple[np.ndarray, int]:
        """Full Hessian 2 R^T W R + a D^T diag(rho''(d)) D, banded.

        The Huber second derivative is 1 in the quadratic branch and 0
        in the linear branch, so this is the natural (piecewise)
        Hessian used by the Newton step and by the implicit-
        differentiation covariance in kc761unfold.errors.
        """
        d = self.d_op @ mu
        hh = huber_hess(d, self.delta)
        omega = (self.d_op.T @ sparse.diags(hh) @ self.d_op).tocsr()
        h = self._rtwr + self.alpha * omega
        return csr_to_upper_banded(h, self.n)

    # -- quadratic warm start ----------------------------------------------

    def warm_start(self) -> np.ndarray:
        """Approximate non-negative quadratic-branch solution.

        The quadratic branch of the Huber objective (|d| <= delta) is
        chi2 + (a/2) |D mu|^2, whose normal equations are
        (2 R^T W R + a D^T D) mu = 2 R^T W y; a few banded
        factorizations with every negative bin dropped per iteration
        land close to its non-negative optimum, from which the Newton
        iterations polish the Huber solution.
        """
        q = (self.d_op.T @ self.d_op).tocsr()
        h2 = self._rtwr + self.alpha * q
        ab, u = csr_to_upper_banded(h2, self.n)
        b = 2.0 * (self.r.T @ (self.w * self.y))
        free = np.ones(self.n, dtype=bool)
        mu = np.zeros(self.n)
        for _ in range(50):
            factor = embed_and_factor(ab, u, free)
            mu = solve_embedded(ab, u, free, b, factor=factor)
            viol = free & (mu < 0.0)
            if not viol.any():
                break
            free = free & ~viol
        return np.where(free, mu, 0.0)

    # -- damped Newton with an active set ------------------------------------

    def solve(self, maxiter: int = 200) -> np.ndarray:
        """Minimize the SWR objective with banded damped Newton steps.

        The active set is the positive bins plus the bins whose gradient
        wants to enter; a backtracking line search with clipping at 0
        enforces the bound.  Convergence when the projected gradient on
        the free set reaches the tolerance.
        """
        mu = self.warm_start()
        nit = 0
        for nit in range(1, maxiter + 1):
            g = self.grad_mu(mu)
            scale = max(1.0, float(np.abs(g).max()))
            free = (mu > 0.0) | (g < -1e-12 * scale)
            ab, u = self.hessian_banded(mu)
            factor = embed_and_factor(ab, u, free)
            step = solve_embedded(ab, u, free, -g, factor=factor)
            dg = float(g @ step)
            if dg >= 0.0 or not np.isfinite(dg):
                break
            t = 1.0
            loss0 = self._loss(mu)
            for _ in range(60):
                mu_new = np.maximum(mu + t * step, 0.0)
                if self._loss(mu_new) <= loss0 + 1e-4 * t * dg:
                    break
                t *= 0.5
            mu = mu_new
            if np.abs(g[free]).max() <= 1e-8 * scale:
                break
        clip_tol = max(1e-6, 1e-9 * float(np.max(mu)))
        self.mu = np.where(mu < clip_tol, 0.0, mu)
        self.free = self.mu > 0.0
        self.chi2 = self._chi2(self.mu)
        self.pen_cost = float(huber_value(self.d_op @ self.mu,
                                          self.delta).sum())
        self.n_iter = nit
        return self.mu
