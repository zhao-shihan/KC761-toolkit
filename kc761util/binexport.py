"""Shared writer for the toolkit's temporary binary export files.

Each exporter writes a magic line, then typed primitives and (optionally)
id-tagged float64 payload blocks in a fixed order; the matching ROOT
macro reads them back and verifies the block ids, so a layout drift fails
loudly instead of silently swapping same-shaped blocks.  The file is
created in the OS temp directory and left in place on success (the macro
deletes it after a successful conversion); on a Python exception it is
removed, so callers can keep it for inspection after a macro failure.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


class ExportWriter:
    """One temporary binary export file (use as a context manager)."""

    def __init__(self, prefix: str, magic: bytes):
        fd, self.path = tempfile.mkstemp(prefix=prefix, suffix=".tmp")
        self._fh = os.fdopen(fd, "wb")
        self._fh.write(magic)

    def __enter__(self) -> "ExportWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._fh.close()
        if exc_type is not None:
            try:
                os.unlink(self.path)
            except OSError:
                pass
        return False

    def put_i64(self, value: int) -> None:
        self._fh.write(np.int64(value).tobytes())

    def put_f64(self, values) -> None:
        self._fh.write(np.asarray(values, dtype=np.float64).tobytes())

    def put_text(self, text: str) -> None:
        self._fh.write(str(text).encode("ascii", errors="replace") + b"\n")

    def put_block(self, block_id: int, values) -> None:
        """One id-tagged float64 payload block."""
        self.put_i64(block_id)
        self.put_f64(values)
