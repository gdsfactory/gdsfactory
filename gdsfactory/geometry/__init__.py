"""Geometric operations: Booleans, DRC checks."""

from __future__ import annotations

from gdsfactory.geometry import functions
from gdsfactory.geometry.boolean import boolean
from gdsfactory.geometry.boolean_klayout import boolean_klayout
from gdsfactory.geometry.boolean_polygons import boolean_polygons
from gdsfactory.geometry.fillet import fillet
from gdsfactory.geometry.invert import invert
from gdsfactory.geometry.layer_priority import layer_priority
from gdsfactory.geometry.manhattanize import manhattanize_polygon
from gdsfactory.geometry.offset import offset
from gdsfactory.geometry.outline import outline
from gdsfactory.geometry.trim import trim
from gdsfactory.geometry.union import union
from gdsfactory.geometry.xor_diff import xor_diff

__all__ = (
    "boolean",
    "boolean_klayout",
    "boolean_polygons",
    "fillet",
    "functions",
    "invert",
    "layer_priority",
    "manhattanize_polygon",
    "offset",
    "outline",
    "trim",
    "union",
    "xor_diff",
)
