"""CD SEM structures."""

from __future__ import annotations

__all__ = ["cdsem_straight_density", "gaps", "widths"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.pcms.cdsem_straight_density import gaps, widths
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats


@gf.cell_with_module_name(tags=["pcms"])
def cdsem_straight_density(
    widths: Floats = widths,
    gaps: Floats = gaps,
    length: float = 420.0,
    label: str = "",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec | None = "text_rectangular",
    text_size: float = 1.0,
) -> Component:
    """Returns sweep of dense straight lines.

    Args:
        widths: list of widths.
        gaps: list of gaps.
        length: of the lines.
        label: defaults to widths[0] gaps[0].
        cross_section: spec.
        text: optional function for text.
        text_size: size of the text.
    """
    return cf.cdsem_straight_density(
        widths=widths,
        gaps=gaps,
        length=length,
        label=label,
        cross_section=cross_section,
        text=text,
        text_size=text_size,
    )
