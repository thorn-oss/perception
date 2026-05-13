"""Helpers for importing optional dependencies with friendly error messages."""

from __future__ import annotations

import importlib
from typing import Any


def import_optional(package: str, *, extra: str) -> Any:
    """Import ``package`` or raise ``ImportError`` pointing at ``extra``.

    Returns the imported module typed as ``Any`` so that attribute access
    (including in type annotations like ``pd.DataFrame``) is not flagged
    by static type checkers when the optional package is not installed in
    the type-checking environment.

    Parameters
    ----------
    package:
        The importable module name (e.g. ``"faiss"``, ``"pandas"``).
    extra:
        The name of the ``perception`` extra that provides ``package``
        (e.g. ``"approximate-deduplication"``).
    """
    try:
        return importlib.import_module(package)
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without extra installed
        raise ImportError(
            f"`{package}` is required for this functionality. Install the "
            f"'{extra}' extra with `pip install perception[{extra}]`."
        ) from exc
