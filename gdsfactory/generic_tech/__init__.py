"""Deprecated module - use gdsfactory.gpdk instead.

This module provides backward compatibility for code using the old
generic_tech module name. All functionality has moved to gdsfactory.gpdk.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from gdsfactory.gpdk import *

# Issue deprecation warning on import
warnings.warn(
    "The 'gdsfactory.generic_tech' module is deprecated and will be removed in a future version. "
    "Please update your imports to use 'gdsfactory.gpdk' instead:\n"
    "  from gdsfactory.gpdk import LAYER, LAYER_STACK, get_generic_pdk\n"
    "Or for submodules:\n"
    "  from gdsfactory.gpdk.layer_map import LAYER\n"
    "  from gdsfactory.gpdk.layer_stack import LAYER_STACK",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LAYER",
    "LAYER_CONNECTIVITY",
    "LAYER_STACK",
    "PDK",
    "PORT_LAYER_TO_TYPE",
    "PORT_MARKER_LAYER_TO_TYPE",
    "PORT_TYPE_TO_MARKER_LAYER",
    "get_generic_pdk",
]


def __getattr__(name: str) -> Any:
    """Lazy import attributes from gdsfactory.gpdk."""
    if name in __all__:
        import gdsfactory.gpdk as gpdk

        return getattr(gpdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from gdsfactory.gpdk import (
        LAYER,
        LAYER_CONNECTIVITY,
        LAYER_STACK,
        PDK,
        PORT_LAYER_TO_TYPE,
        PORT_MARKER_LAYER_TO_TYPE,
        PORT_TYPE_TO_MARKER_LAYER,
        get_generic_pdk,
    )
