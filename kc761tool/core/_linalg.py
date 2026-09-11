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

from kc761tool.errors import SolverError

DENSE_LIMIT = 512
"""Systems at or below this size use a dense Cholesky factorization."""

BAND_FRACTION = 4
"""A banded factorization is used when the half-bandwidth is below n / this."""

DENSE_NORMAL_LIMIT = 8_000_000
"""Dense ``R^T W R`` entries above which the sparse product is preferred."""

DENSE_NORMAL_DENSITY = 0.05
"""Minimum ``nnz / (rows * cols)`` for the dense product to be worthwhile."""


def csr_from_column_triples(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    values: NDArray[np.float64],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    """Build a CSR matrix from column-sorted, duplicate-free triples.

    The kernel pattern is grouped by deposition column with ascending channel
    rows, so a CSC matrix can be filled directly (no scipy COO sort) and then
    converted; this is ~1.4x faster than ``csr_matrix((values, (rows, cols)))``
    at the 2048 scale (D-174).
    """
    counts = np.bincount(cols, minlength=shape[1]).astype(np.int64)
    indptr = np.zeros(shape[1] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.ascontiguousarray(rows, dtype=np.int32)
    return sparse.csc_matrix(
        (np.asarray(values, dtype=np.float64), indices, indptr), shape=shape
    ).tocsr()


def prefer_dense_factor(matrix: sparse.spmatrix) -> bool:
    """Whether the Cholesky factorization should run on a dense matrix (D-172).

    Dense systems (or systems whose stored sparsity exceeds 20%) factor faster
    through LAPACK. Exposed so the active-set solver can densify once and then
    index with ``numpy.ix_`` instead of paying sparse fancy-indexing per
    iteration while keeping exactly one definition of the policy.
    """
    size = int(matrix.shape[0])
    return size <= DENSE_LIMIT or matrix.nnz > 0.2 * size * size


def factor_dense_spd(dense: NDArray[np.float64]) -> SpdFactor:
    """Cholesky-factor an already-dense symmetric positive-definite matrix."""
    symmetric = 0.5 * (dense + dense.T)
    try:
        factor = linalg.cho_factor(symmetric, lower=True, check_finite=False)
    except linalg.LinAlgError as exc:
        raise SolverError(f"normal matrix is not positive definite: {exc}") from exc
    return SpdFactor("dense", dense=factor)


def prefer_dense_normal(matrix: sparse.spmatrix) -> bool:
    """Whether ``matrix.T @ matrix`` is cheaper through dense BLAS (D-172).

    The normal-equation product is the same either way; the dense path avoids
    scipy's sparse-sparse symbolic pass (``csr_matmat_maxnnz``) and uses BLAS-3,
    while the sparse path avoids materializing a dense operand. The threshold is
    a measured crossover: below ``DENSE_NORMAL_LIMIT`` entries and above
    ``DENSE_NORMAL_DENSITY`` the dense product wins on the 2048 problem.
    """
    rows, cols = matrix.shape
    entries = int(rows) * int(cols)
    if entries == 0:
        return True
    return entries <= DENSE_NORMAL_LIMIT and matrix.nnz > DENSE_NORMAL_DENSITY * entries


def weighted_normal(
    matrix: sparse.spmatrix, weights: NDArray[np.float64]
) -> sparse.csr_matrix:
    """Return ``matrix.T @ diag(weights) @ matrix`` (half-Hessian ``A``).

    Uses dense BLAS when :func:`prefer_dense_normal` holds and falls back to the
    sparse product otherwise. The result is symmetrized so downstream Cholesky
    sees an exactly symmetric matrix regardless of the path.
    """
    hessian, _rhs = weighted_normal_and_rhs(matrix, weights, None)
    return hessian


def weighted_normal_and_rhs(
    matrix: sparse.spmatrix,
    weights: NDArray[np.float64],
    rhs: NDArray[np.float64] | None,
) -> tuple[sparse.csr_matrix, NDArray[np.float64] | None]:
    """Return ``(matrix.T diag(weights) matrix, matrix.T (weights * rhs))``.

    One dense materialization serves both products, so the gradient offset
    ``b = R^T W y`` is free once the half-Hessian is built. ``rhs`` may be
    ``None`` for callers that only need the matrix.
    """
    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    cisr = matrix.tocsr().astype(np.float64)
    if cisr.shape[0] != weights_array.size:
        raise ValueError(
            f"weights has {weights_array.size} entries, matrix has {cisr.shape[0]} rows"
        )
    rhs_array: NDArray[np.float64] | None = None
    if rhs is not None:
        rhs_array = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs_array.size != cisr.shape[0]:
            raise ValueError(
                f"rhs has {rhs_array.size} entries, matrix has {cisr.shape[0]} rows"
            )
    if prefer_dense_normal(cisr):
        dense = cisr.toarray()
        hessian = (dense.T * weights_array) @ dense
        hessian = 0.5 * (hessian + hessian.T)
        gradient = None if rhs_array is None else dense.T @ (weights_array * rhs_array)
        return sparse.csr_matrix(hessian), gradient
    weighted = cisr.T @ sparse.diags(weights_array)
    hessian = (weighted @ cisr).tocsr()
    gradient = None if rhs_array is None else np.asarray(weighted @ rhs_array, dtype=np.float64)
    return hessian, gradient


class SpdFactor:
    """Reusable factorization of a symmetric positive-definite matrix (D-172).

    The policy matches :func:`factor_spd`. A factor may be applied to several
    right-hand sides (a matrix or a stream of vectors) without refactorizing,
    which the uncertainty propagation needs for its many reduced solves.
    """

    __slots__ = ("_kind", "_dense", "_banded", "_sparse")

    def __init__(
        self,
        kind: str,
        *,
        dense: object | None = None,
        banded: object | None = None,
        sparse_factor: object | None = None,
    ) -> None:
        self._kind = kind
        self._dense = dense
        self._banded = banded
        self._sparse = sparse_factor

    def solve(self, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``matrix x = rhs`` with the stored factor."""
        if self._kind == "dense":
            return np.asarray(
                linalg.cho_solve(self._dense, rhs, check_finite=False), dtype=np.float64
            )
        if self._kind == "banded":
            return np.asarray(
                linalg.cho_solve_banded(self._banded, rhs, check_finite=False),
                dtype=np.float64,
            )
        return np.asarray(self._sparse.solve(rhs), dtype=np.float64)


def factor_spd(matrix: sparse.spmatrix) -> SpdFactor:
    """Factorize a symmetric positive-definite matrix with the shared policy.

    Dense Cholesky for small or dense systems, banded Cholesky when the
    half-bandwidth is below ``n / BAND_FRACTION``, sparse LU otherwise. A
    non-positive-definite or singular matrix raises
    :class:`kc761tool.errors.SolverError`; there is no silent regularization.
    """
    if not sparse.issparse(matrix):
        raise SolverError("normal matrix must be a scipy sparse matrix")
    sub = matrix.tocsc()
    size = int(sub.shape[0])
    if sub.shape[0] != sub.shape[1] or size == 0:
        raise SolverError(f"normal matrix must be non-empty and square, got {sub.shape}")
    if prefer_dense_factor(sub):
        return factor_dense_spd(sub.toarray())
    bandwidth = half_bandwidth(sub)
    if bandwidth * BAND_FRACTION < size:
        banded = to_banded(sub, bandwidth)
        try:
            factor = linalg.cholesky_banded(banded, lower=True, check_finite=False)
        except linalg.LinAlgError as exc:
            raise SolverError(f"banded Cholesky failed: {exc}") from exc
        return SpdFactor("banded", banded=(factor, True))
    try:
        sparse_factor = splinalg.splu(sub)
    except RuntimeError as exc:
        raise SolverError(f"sparse factorization failed: {exc}") from exc
    return SpdFactor("sparse", sparse_factor=sparse_factor)


def solve_spd(matrix: sparse.spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``matrix x = rhs`` for a symmetric positive-definite matrix.

    One-shot form of :func:`factor_spd`; callers that solve repeatedly should
    keep the :class:`SpdFactor` instead of refactorizing.
    """
    return factor_spd(matrix).solve(rhs)


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
