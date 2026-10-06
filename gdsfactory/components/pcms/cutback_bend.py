from __future__ import annotations

__all__ = [
    "cutback_bend",
    "cutback_bend90",
    "cutback_bend90circular",
    "cutback_bend180",
    "cutback_bend180circular",
    "staircase",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name(tags=["pcms"])
def cutback_bend(
    component: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 5,
    **kwargs: Any,
) -> Component:
    """We recommend using cutback_bend90 instead for a smaller footprint.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        kwargs: cross_section settings.

        this is a column
            _
          _|
        _|

        _ this is a row
    """
    return cf.cutback_bend(
        component=component,
        straight=straight,
        straight_length=straight_length,
        rows=rows,
        cols=cols,
        **kwargs,
    )


@gf.cell_with_module_name(tags=["pcms"])
def cutback_bend90(
    component: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 6,
    spacing: int = 5,
    **kwargs: Any,
) -> Component:
    """Returns bend90 cutback.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        spacing: in um.
        kwargs: cross_section settings.

           _
        |_| |
    """
    return cf.cutback_bend90(
        component=component,
        straight=straight,
        straight_length=straight_length,
        rows=rows,
        cols=cols,
        spacing=spacing,
        **kwargs,
    )


@gf.cell_with_module_name(tags=["pcms"])
def staircase(
    component: ComponentSpec | Component = "bend_euler",
    straight: ComponentSpec = "straight",
    length_v: float = 5.0,
    length_h: float = 5.0,
    rows: int = 4,
    **kwargs: Any,
) -> Component:
    """Returns staircase.

    Args:
        component: bend spec.
        straight: straight spec.
        length_v: vertical length.
        length_h: vertical length.
        rows: number of rows.
        cols: number of cols.
        kwargs: cross_section settings.
    """
    return cf.staircase(
        component=component,
        straight=straight,
        length_v=length_v,
        length_h=length_h,
        rows=rows,
        **kwargs,
    )


@gf.cell_with_module_name(tags=["pcms"])
def cutback_bend180(
    component: ComponentSpec = "bend_euler180",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 6,
    spacing: float = 3.0,
    **kwargs: Any,
) -> Component:
    """Returns cutback to measure u bend loss.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        spacing: in um.
        kwargs: cross_section settings.

          _
        _| |_  this is a row

        _ this is a column
    """
    return cf.cutback_bend180(
        component=component,
        straight=straight,
        straight_length=straight_length,
        rows=rows,
        cols=cols,
        spacing=spacing,
        **kwargs,
    )


cutback_bend180circular = CellAlias(cutback_bend180, component="bend_circular180")
cutback_bend90circular = CellAlias(cutback_bend90, component="bend_circular")
