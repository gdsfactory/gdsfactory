"""CD SEM structures."""

from __future__ import annotations

__all__ = ["cdsem_straight"]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.pcms.cdsem_straight import LINE_LENGTH
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cdsem_straight(
    widths: Sequence[float] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    length: float = LINE_LENGTH,
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = "text_rectangular",
    spacing: float = 7.0,
    positions: Sequence[float | None] | None = None,
    text_size: float = 1,
) -> Component:
    """Returns straight waveguide lines width sweep.

    Args:
        widths: for the sweep.
        length: for the line.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        text_size: in um.
    """
    return cf.cdsem_straight(
        widths=widths,
        length=length,
        cross_section=cross_section,
        text=text,
        spacing=spacing,
        positions=positions,
        text_size=text_size,
    )
