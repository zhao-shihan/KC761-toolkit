"""Spectrum arithmetic behind ``kc761tool specadd`` and ``kc761tool specsub``.

Two formula IDs live here (``docs/derivations.md``):

* :data:`ADD` (F-SPEC-1) sums two spectrum products: values add, variances add
  and the DAQ times add.
* :data:`SUB` (F-SPEC-2) subtracts the second spectrum scaled by the DAQ-time
  ratio ``r = t_first / t_second``; the output inherits the first DAQ time.

Both operations floor every input bin error at one count before combining, so a
zero-count bin never claims zero uncertainty. ``run_specadd`` and ``run_specsub``
are the entries for the two CLI commands; they load the operand products through
:mod:`kc761tool.schema.io`, validate the shared channel axis bitwise, build the
``spectrum`` product with its provenance and write it atomically. The CLI layer
stays a surface: it parses arguments, resolves the default output name and
reports the result.
"""

from __future__ import annotations

from kc761tool.spectra.combine import (
    ADD,
    SUB,
    SpectrumCombination,
    SpectrumResult,
    combine_spectra,
    run_specadd,
    run_specsub,
)

__all__ = [
    "ADD",
    "SUB",
    "SpectrumCombination",
    "SpectrumResult",
    "combine_spectra",
    "run_specadd",
    "run_specsub",
]
