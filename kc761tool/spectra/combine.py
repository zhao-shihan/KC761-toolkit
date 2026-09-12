"""Spectrum addition (F-SPEC-1) and DAQ-time scaled subtraction (F-SPEC-2).

Both operations share one pipeline: validate the operand pair, floor every input
bin error at one count, combine values and variances, assemble the ``spectrum``
product and write it atomically (F-IO-1). ``docs/derivations.md`` carries the
formulas and their limits.

* F-SPEC-1 (``ADD``): ``values = v_a + v_b``,
  ``variances = max(e_a, 1)**2 + max(e_b, 1)**2``, ``daq_time_s = t_a + t_b``.
* F-SPEC-2 (``SUB``): ``r = t_a / t_b``, ``values = v_a - r v_b``,
  ``variances = max(e_a, 1)**2 + r**2 max(e_b, 1)**2``, ``daq_time_s = t_a``,
  where ``e = sqrt(fSumw2)``.

Preconditions (always on, both operations): both operands are ``spectrum``
products, their channel axes match bitwise (unit and edges) and both DAQ times
are finite and positive. The output ``source_file`` is the audit string
``"<first> <operator> <second>"`` in positional order (D-185); both real input
paths and their sha256 live in ``provenance.inputs``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kc761tool.errors import Kc761toolError, SchemaError, ValidationError
from kc761tool.schema.axes import Axis
from kc761tool.schema.io import (
    build_provenance,
    read_product,
    validate_output_path,
    write_product,
)
from kc761tool.schema.products import (
    SCHEMA_VERSION,
    Histogram1D,
    SpectrumProduct,
    dispatch_key_for,
)

#: Operation tokens (D-185). One token names the operation, the default output
#: connector (``<first>-add-<second>.root`` / ``<first>-sub-<second>.root``) and
#: the command and producer strings (``kc761tool spec<token>``).
ADD: Final = "add"
SUB: Final = "sub"

#: Operator sign of each operation, used only in the ``source_file`` audit string.
OPERATORS: Final[dict[str, str]] = {ADD: "+", SUB: "-"}

#: Every input bin error is floored at this many counts before combining, so a
#: zero-count bin does not claim zero uncertainty (F-SPEC-1/F-SPEC-2).
ERROR_FLOOR_COUNTS: Final = 1.0


@dataclass(frozen=True)
class SpectrumCombination:
    """Result of F-SPEC-1/F-SPEC-2 before it is wrapped in a product."""

    operation: str
    axis: Axis
    values: NDArray[np.float64]
    variances: NDArray[np.float64]
    daq_time_s: float
    scale: float  # factor applied to the second input (1.0 for ADD)

    @property
    def operator(self) -> str:
        """Operator sign of this combination (``+`` for ADD, ``-`` for SUB)."""
        return OPERATORS[self.operation]

    def source_file(self, first_source: str, second_source: str) -> str:
        """Return the audit string ``"<first> <operator> <second>"`` (D-185)."""
        return f"{first_source} {self.operator} {second_source}"


@dataclass(frozen=True)
class SpectrumResult:
    """Outcome of :func:`run_specadd`/:func:`run_specsub`."""

    combination: SpectrumCombination
    product: SpectrumProduct | None
    product_path: Path | None
    first_path: Path | None
    second_path: Path | None


def coerce_spectrum(
    value: str | Path | SpectrumProduct,
    *,
    strict: bool,
    label: str,
) -> tuple[SpectrumProduct, Path | None]:
    """Return a spectrum product and the path it came from (if any)."""
    if isinstance(value, SpectrumProduct):
        return value, None
    path = Path(value).expanduser()
    if not path.is_file():
        raise Kc761toolError(f"{label} spectrum file not found: {path}")
    product = read_product(path, strict=strict)
    if not isinstance(product, SpectrumProduct):
        raise SchemaError(f"{path}: expected a spectrum product, got {dispatch_key_for(product)!r}")
    return product, path


def check_operands(first: SpectrumProduct, second: SpectrumProduct) -> tuple[float, float]:
    """Validate the F-SPEC-1/F-SPEC-2 preconditions; return both DAQ times.

    The shared channel axis must match bitwise (unit and edges) and both DAQ
    times must be finite and positive. A violation is a ``ValidationError`` in
    every mode (no clamping, no NaN substitution).
    """
    if first.spectrum.axis.unit != second.spectrum.axis.unit:
        raise ValidationError(
            "operand channel unit mismatch: "
            f"{first.spectrum.axis.unit!r} vs {second.spectrum.axis.unit!r}"
        )
    if not np.array_equal(first.spectrum.axis.edges, second.spectrum.axis.edges):
        raise ValidationError("operand binning mismatch: the channel axes differ")
    times: list[float] = []
    for label, product in (("first", first), ("second", second)):
        daq_time = float(product.daq_time_s)
        if not np.isfinite(daq_time) or daq_time <= 0.0:
            raise ValidationError(
                f"{label} operand daq_time_s must be positive and finite (got {daq_time!r})"
            )
        times.append(daq_time)
    return times[0], times[1]


def combine_spectra(
    first: SpectrumProduct,
    second: SpectrumProduct,
    *,
    operation: str,
) -> SpectrumCombination:
    """Apply F-SPEC-1 (``operation=ADD``) or F-SPEC-2 (``operation=SUB``)."""
    if operation not in OPERATORS:
        raise ValidationError(
            f"unknown spectrum operation {operation!r}; expected one of {sorted(OPERATORS)}"
        )
    first_time, second_time = check_operands(first, second)
    first_values = np.asarray(first.spectrum.values, dtype=np.float64)
    second_values = np.asarray(second.spectrum.values, dtype=np.float64)
    first_error = np.maximum(
        np.sqrt(np.asarray(first.spectrum.variances, dtype=np.float64)), ERROR_FLOOR_COUNTS
    )
    second_error = np.maximum(
        np.sqrt(np.asarray(second.spectrum.variances, dtype=np.float64)), ERROR_FLOOR_COUNTS
    )
    if operation == ADD:
        scale = 1.0
        values = first_values + second_values
        variances = first_error**2 + second_error**2
        daq_time_s = first_time + second_time
    else:
        scale = first_time / second_time
        values = first_values - scale * second_values
        variances = first_error**2 + scale**2 * second_error**2
        daq_time_s = first_time
    return SpectrumCombination(
        operation=operation,
        axis=first.spectrum.axis,
        values=values,
        variances=variances,
        daq_time_s=daq_time_s,
        scale=scale,
    )


def run_specadd(
    first: str | Path | SpectrumProduct,
    second: str | Path | SpectrumProduct,
    *,
    output: str | Path | None = None,
    force: bool = False,
    strict: bool = False,
    command: str = "kc761tool specadd",
    arguments: Sequence[tuple[str, str]] = (),
    producer: str = "kc761tool-specadd",
) -> SpectrumResult:
    """Sum two spectrum products (F-SPEC-1) and optionally write the product."""
    return _run_combination(
        ADD,
        first,
        second,
        output=output,
        force=force,
        strict=strict,
        command=command,
        arguments=arguments,
        producer=producer,
    )


def run_specsub(
    first: str | Path | SpectrumProduct,
    second: str | Path | SpectrumProduct,
    *,
    output: str | Path | None = None,
    force: bool = False,
    strict: bool = False,
    command: str = "kc761tool specsub",
    arguments: Sequence[tuple[str, str]] = (),
    producer: str = "kc761tool-specsub",
) -> SpectrumResult:
    """Subtract the second spectrum scaled by DAQ time (F-SPEC-2) and write it."""
    return _run_combination(
        SUB,
        first,
        second,
        output=output,
        force=force,
        strict=strict,
        command=command,
        arguments=arguments,
        producer=producer,
    )


def _run_combination(
    operation: str,
    first: str | Path | SpectrumProduct,
    second: str | Path | SpectrumProduct,
    *,
    output: str | Path | None,
    force: bool,
    strict: bool,
    command: str,
    arguments: Sequence[tuple[str, str]],
    producer: str,
) -> SpectrumResult:
    if output is not None:
        # D-171: validate the target before reading the operands.
        validate_output_path(output, force=force)
    first_product, first_path = coerce_spectrum(first, strict=strict, label="first")
    second_product, second_path = coerce_spectrum(second, strict=strict, label="second")
    combination = combine_spectra(first_product, second_product, operation=operation)

    product: SpectrumProduct | None = None
    product_path: Path | None = None
    if output is not None:
        product = SpectrumProduct(
            format_version=SCHEMA_VERSION,
            spectrum=Histogram1D(
                axis=combination.axis,
                values=combination.values,
                variances=combination.variances,
            ),
            daq_time_s=combination.daq_time_s,
            source_file=combination.source_file(
                _source_label(first_product, first_path),
                _source_label(second_product, second_path),
            ),
            provenance=build_provenance(
                producer=producer,
                command=command,
                arguments=tuple(arguments),
                inputs=[path for path in (first_path, second_path) if path is not None],
            ),
        )
        product_path = write_product(product, output, force=force, strict=strict)

    return SpectrumResult(
        combination=combination,
        product=product,
        product_path=product_path,
        first_path=first_path,
        second_path=second_path,
    )


def _source_label(product: SpectrumProduct, path: Path | None) -> str:
    """Audit label of one operand: its path, or its own ``source_file`` in memory."""
    return str(path) if path is not None else product.source_file


__all__ = [
    "ADD",
    "ERROR_FLOOR_COUNTS",
    "OPERATORS",
    "SUB",
    "SpectrumCombination",
    "SpectrumResult",
    "check_operands",
    "coerce_spectrum",
    "combine_spectra",
    "run_specadd",
    "run_specsub",
]
