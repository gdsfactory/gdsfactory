"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import warnings

from kfactory import cell

cell_without_validator = cell


def clear_cache() -> None:
    """Clear the cache of the cell decorator."""
    warnings.warn("clear_cache is deprecated and does nothing in klayout backend.")


__all__ = ["cell", "cell_without_validator"]
