"""Shared linear-algebra policies for the numeric core (D-60).

The non-negative solver and the uncertainty propagation must factorize the
same positive-definite normal matrix with the same fallback ladder; keeping
the policy here prevents the two call sites from drifting apart.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg

from kc761.errors import SolverError

DENSE_LIMIT = 512
"""Systems at or below this size use a dense Cholesky factorization."""

BAND_FRACTION = 4
"""A banded factorization is used when the half-bandwidth is below n / this."""


def solve_spd(matrix: sparse.spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``matrix x = rhs`` for a symmetric positive-definite matrix.

    Policy: dense Cholesky for small or dense systems, banded Cholesky when
    the half-bandwidth is below ``n / BAND_FRACTION``, sparse LU otherwise.
    A non-positive-definite or singular matrix raises
    :class:`kc761.errors.SolverError`; there is no silent regularization.
    """
    if not sparse.issparse(matrix):
        raise SolverError("normal matrix must be a scipy sparse matrix")
    sub = matrix.tocsc()
    size = int(sub.shape[0])
    if sub.shape[0] != sub.shape[1] or size == 0:
        raise SolverError(f"normal matrix must be non-empty and square, got {sub.shape}")
    if size <= DENSE_LIMIT or sub.nnz > 0.2 * size * size:
        dense = sub.toarray()
        dense = 0.5 * (dense + dense.T)
        try:
            factor = linalg.cho_factor(dense, lower=True, check_finite=False)
        except linalg.LinAlgError as exc:
            raise SolverError(f"normal matrix is not positive definite: {exc}") from exc
        return np.asarray(linalg.cho_solve(factor, rhs, check_finite=False), dtype=np.float64)
    bandwidth = half_bandwidth(sub)
    if bandwidth * BAND_FRACTION < size:
        banded = to_banded(sub, bandwidth)
        try:
            return np.asarray(
                linalg.solveh_banded(banded, rhs, lower=True, check_finite=False),
                dtype=np.float64,
            )
        except linalg.LinAlgError as exc:
            raise SolverError(f"banded Cholesky failed: {exc}") from exc
    try:
        factor = splinalg.splu(sub)
    except RuntimeError as exc:
        raise SolverError(f"sparse factorization failed: {exc}") from exc
    return np.asarray(factor.solve(rhs), dtype=np.float64)


def half_bandwidth(matrix: sparse.spmatrix) -> int:
    """Largest ``|row - column|`` of the nonzero pattern."""
    rows, cols = matrix.nonzero()
    if rows.size == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


def to_banded(matrix: sparse.spmatrix, bandwidth: int) -> NDArray[np.float64]:
    """Convert the lower triangle to the ``solveh_banded(lower=True)`` layout."""
    rows, cols = matrix.nonzero()
    values = np.asarray(matrix[rows, cols]).ravel()
    banded = np.zeros((bandwidth + 1, matrix.shape[0]), dtype=np.float64)
    lower = rows >= cols
    np.add.at(banded, (rows[lower] - cols[lower], cols[lower]), values[lower])
    return banded
