"""Cell decorator for functions that return a Component."""
from __future__ import annotations

from kfactory import cell

cell_without_validator = cell

__all__ = ["cell", "cell_without_validator"]
