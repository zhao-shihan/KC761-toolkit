"""Always-on validation helpers shared by the numeric core (D-62).

These checks run in every mode and cannot be disabled. They raise
:class:`kc761.errors.ValidationError` with an actionable message; they never
clamp, substitute NaN or fall back to defaults (AGENTS hard rule 12).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from kc761.errors import ValidationError


def as_float_array(name: str, value: object, *, ndim: int | None = None) -> NDArray[np.float64]:
    """Convert ``value`` to ``float64`` and require the requested dimensionality."""
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValidationError(f"{name} contains non-finite values")
    return array


def require_positive(name: str, value: float) -> float:
    """Require a finite, strictly positive scalar."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def require_same_length(name: str, left: object, right: object) -> None:
    """Require two one-dimensional arrays to share their length."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValidationError(
            f"{name} shape mismatch: {left_array.shape} vs {right_array.shape}"
        )


def check_response_matrix(response: sparse.spmatrix) -> sparse.csr_matrix:
    """Validate and return a CSR response matrix (shape, finiteness, sign)."""
    if not sparse.issparse(response):
        raise ValidationError("response must be a scipy sparse matrix")
    matrix = response.tocsr().astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValidationError(f"response has invalid shape {matrix.shape}")
    if not np.isfinite(matrix.data).all():
        raise ValidationError("response contains non-finite values")
    if np.any(matrix.data < 0.0):
        raise ValidationError("response must be non-negative")
    return matrix
