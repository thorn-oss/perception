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
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - exercised only without extra installed
        # Only convert to the friendly "install the extra" hint when the
        # missing module is the optional package itself (or a submodule of
        # it). If the optional package is installed but one of its own
        # imports failed, re-raise the original error so the real cause is
        # not hidden.
        missing = exc.name or ""
        top_level = package.split(".", 1)[0]
        if missing != top_level and not missing.startswith(top_level + "."):
            raise
        raise ImportError(
            f"`{package}` is required for this functionality. Install the "
            f"'{extra}' extra with `pip install perception[{extra}]`."
        ) from exc
