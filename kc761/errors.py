"""Exception hierarchy for the KC761 toolkit.

Everything the toolkit raises derives from :class:`Kc761Error`, so the CLI can
convert any failure into a single ``[kc761.<command>] error: ...`` message with
a stable exit code (docs/plan.md D-67/D-68): 0 success, 1 runtime failure,
2 usage error.

The always-on validation layer raises :class:`SchemaError` (product contract,
version, axes, shapes) or :class:`ValidationError` (values); strict-mode
certificates raise :class:`CertificateError` and report the formula ID
(docs/derivations.md).
"""

from __future__ import annotations


class Kc761Error(Exception):
    """Base class for every error raised by the toolkit."""


class UsageError(Kc761Error):
    """Invalid invocation or argument combination (exit code 2)."""


class SchemaError(Kc761Error):
    """A product file violates the frozen schema."""


class ValidationError(Kc761Error):
    """A valid-shaped input violates a numerical or physical invariant."""


class ProvenanceError(Kc761Error):
    """Provenance metadata is missing, malformed or inconsistent."""


class UnsupportedError(Kc761Error):
    """The request is outside the frozen support envelope."""


class SolverError(Kc761Error):
    """A numerical solver failed to produce a solution."""


class CertificateError(Kc761Error):
    """A strict-mode runtime certificate failed."""

    def __init__(self, formula_id: str, message: str) -> None:
        super().__init__(f"[{formula_id}] {message}")
        self.formula_id = formula_id
