"""Product IO contract: atomic write, reopen validation and overwrite policy.

Formula ID F-IO-1 (docs/derivations.md). Frozen protocol (docs/formats.md):

1. refuse an existing target unless ``force`` is set (D-17);
2. write to ``<target>.part``;
3. close, reopen and validate every object and the ``meta`` tree;
4. rename onto the target (atomic on the same filesystem) (D-16);
5. on any failure remove the partial file and raise.

All product reading and writing goes through this module (AGENTS.md rule 10);
callers must not open product files themselves. Implementation starts in W2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from kc761.schema.products import Product, Provenance

PART_SUFFIX = ".part"


def build_provenance(
    *,
    producer: str,
    command: str,
    arguments: Sequence[tuple[str, str]],
    inputs: Sequence[str | Path],
    extra_dependencies: Sequence[str] = (),
) -> Provenance:
    """Assemble the complete provenance record for one run (F-IO-1/D-18)."""
    raise NotImplementedError("F-IO-1 build_provenance is implemented in W2")


def refuse_overwrite(path: str | Path, *, force: bool = False) -> None:
    """Raise when ``path`` exists and ``force`` is not set (F-IO-1/D-17)."""
    raise NotImplementedError("F-IO-1 refuse_overwrite is implemented in W2")


def atomic_write(
    target: str | Path,
    writer: Callable[[Path], None],
    *,
    verify: Callable[[Path], None],
) -> Path:
    """Write, verify and atomically move a product into place (F-IO-1/D-16).

    ``writer`` receives the ``.part`` path; ``verify`` reopens it and checks
    the full contract. The partial file is removed on failure.
    """
    raise NotImplementedError("F-IO-1 atomic_write is implemented in W2")


def write_product(product: Product, path: str | Path, *, force: bool = False) -> Path:
    """Persist one product atomically and return its final path (F-IO-1)."""
    raise NotImplementedError("F-IO-1 write_product is implemented in W2")


def read_product(path: str | Path) -> Product:
    """Load and validate a product, dispatching on its ``meta`` tree (F-IO-1)."""
    raise NotImplementedError("F-IO-1 read_product is implemented in W2")


def verify_product(path: str | Path) -> None:
    """Reopen a product and validate schema, versions, axes and metadata (F-IO-1)."""
    raise NotImplementedError("F-IO-1 verify_product is implemented in W2")
