from __future__ import annotations

__all__ = ["wafer"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.dies.wafer import _cols_200mm_wafer
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name(tags=["dies"])
def wafer(
    reticle: ComponentSpec = "die",
    cols: tuple[int, ...] = _cols_200mm_wafer,
    xspacing: float | None = None,
    yspacing: float | None = None,
    die_name_col_row: bool = False,
) -> Component:
    """Returns complete wafer. Useful for mask aligner steps.

    Args:
        reticle: spec for each wafer reticle.
        cols: how many columns per row.
        xspacing: optional spacing, defaults to reticle.xsize.
        yspacing: optional spacing, defaults to reticle.ysize.
        die_name_col_row: if True, die name is row_col, otherwise is a number
    """
    return cf.wafer(
        reticle=reticle,
        cols=cols,
        xspacing=xspacing,
        yspacing=yspacing,
        die_name_col_row=die_name_col_row,
    )
