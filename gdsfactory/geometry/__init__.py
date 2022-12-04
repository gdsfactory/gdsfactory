"""Geometric operations: Booleans, DRC checks."""

from __future__ import annotations

from gdsfactory.geometry import functions
from gdsfactory.geometry.boolean import boolean
from gdsfactory.geometry.boolean_klayout import boolean_klayout
from gdsfactory.geometry.check_duplicated_cells import check_duplicated_cells
from gdsfactory.geometry.check_exclusion import check_exclusion
from gdsfactory.geometry.check_inclusion import check_inclusion
from gdsfactory.geometry.check_space import check_space
from gdsfactory.geometry.check_width import check_width
from gdsfactory.geometry.invert import invert
from gdsfactory.geometry.offset import offset
from gdsfactory.geometry.outline import outline
from gdsfactory.geometry.trim import trim
from gdsfactory.geometry.union import union
from gdsfactory.geometry.xor_diff import xor_diff

__all__ = (
    "boolean",
    "boolean_klayout",
    "check_duplicated_cells",
    "check_exclusion",
    "check_inclusion",
    "check_space",
    "check_width",
    "invert",
    "offset",
    "outline",
    "union",
    "xor_diff",
    "functions",
    "trim",
)
