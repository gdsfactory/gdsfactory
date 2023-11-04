"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import warnings

from kfactory import cell

cell_without_validator = cell

warnings.warn(
    "gdsfactory.cell is deprecated and will removed in future versions of gdsfactory."
)


def clear_cache() -> None:
    """Clear the cache of the cell decorator."""
    warnings.warn("clear_cache is deprecated and does nothing in gdsfactory>=8.0.0")


__all__ = ["cell", "cell_without_validator"]
