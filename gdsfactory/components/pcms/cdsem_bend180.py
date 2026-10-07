"""CD SEM structures."""

from __future__ import annotations

__all__ = ["cdsem_bend180"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.pcms.cdsem_bend180 import LINE_LENGTH
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cdsem_bend180(
    width: float = 0.5,
    radius: float = 10.0,
    wg_length: float | None = LINE_LENGTH,
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec = "text_rectangular",
    text_size: float = 1.0,
) -> Component:
    """Returns CDSEM structures.

    Args:
        width: of the line.
        radius: um.
        wg_length: in um.
        straight: spec.
        bend90: spec.
        cross_section: spec.
        text: spec.
        text_size: um.
    """
    return cf.cdsem_bend180(
        width=width,
        radius=radius,
        wg_length=wg_length,
        straight=straight,
        bend90=bend90,
        cross_section=cross_section,
        text=text,
        text_size=text_size,
    )
