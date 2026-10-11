"""CD SEM structures."""

from __future__ import annotations

__all__ = ["cdsem_coupler"]

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cdsem_coupler(
    length: float = 420.0,
    gaps: Sequence[float] = (0.15, 0.2, 0.25),
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = "text_rectangular",
    spacing: float = 7.0,
    positions: Sequence[float | None] | None = None,
    width: float | None = None,
    text_size: float = 1.0,
) -> Component:
    """Returns 2 coupled waveguides gap sweep.

    Args:
        length: for the line.
        gaps: list of gaps for the sweep.
        cross_section: for the lines.
        text: optional text for labels.
        spacing: Optional center to center spacing.
        positions: Optional positions for the text labels.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        text_size: size of the text.
    """
    return cf.cdsem_coupler(
        length=length,
        gaps=gaps,
        cross_section=cross_section,
        text=text,
        spacing=spacing,
        positions=positions,
        width=width,
        text_size=text_size,
    )
