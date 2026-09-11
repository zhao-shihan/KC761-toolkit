"""Runtime configuration: logging and strict-mode resolution.

Strict mode enables the full runtime certificate suites; it is requested with
``--strict`` or ``KC761TOOL_STRICT=1`` (the CLI flag wins). Basic schema, shape and
finiteness validation is always on and is not controlled here
(docs/plan.md D-61/D-62).

Logging uses the standard library with the frozen format
``[kc761tool.<command>] level: message`` (D-67).
"""

from __future__ import annotations

import logging
import os

from kc761tool.errors import UsageError

STRICT_ENV_VAR = "KC761TOOL_STRICT"
LOG_LEVELS = ("debug", "info", "warning", "error")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


class _PrefixFormatter(logging.Formatter):
    """Render ``[kc761tool.<command>] level: message`` on one line."""

    def format(self, record: logging.LogRecord) -> str:
        return f"[{record.name}] {record.levelname.lower()}: {record.getMessage()}"


def strict_enabled(cli_flag: bool = False) -> bool:
    """Resolve strict mode from the CLI flag and ``KC761TOOL_STRICT``.

    Raises:
        UsageError: if the environment variable is set to a non-boolean value.
    """
    if cli_flag:
        return True
    raw = os.environ.get(STRICT_ENV_VAR)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise UsageError(
        f"{STRICT_ENV_VAR} must be a boolean value "
        f"(one of {sorted(_TRUTHY | _FALSY)}), got {raw!r}"
    )


def log_level(name: str) -> int:
    """Map a CLI log-level name to the corresponding logging level."""
    if name not in LOG_LEVELS:
        raise UsageError(f"unknown log level {name!r}; expected one of {LOG_LEVELS}")
    return int(getattr(logging, name.upper()))


def configure_logging(command: str, level: str = "info") -> logging.Logger:
    """Return the ``kc761tool.<command>`` logger with one stderr handler.

    Idempotent: repeated calls for the same command reuse the handler.
    """
    logger = logging.getLogger(f"kc761tool.{command}")
    logger.setLevel(log_level(level))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_PrefixFormatter())
        logger.addHandler(handler)
    return logger
