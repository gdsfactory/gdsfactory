"""CD SEM structures."""

from __future__ import annotations

__all__ = ["cdsem_all"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["pcms"])
def cdsem_all(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    dense_lines_width: float | None = 0.3,
    dense_lines_width_difference: float = 20e-3,
    dense_lines_gap: float = 0.3,
    dense_lines_labels: tuple[str, ...] = ("DL", "DM", "DH"),
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec | None = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec = "text_rectangular",
    spacing: float = 5,
    cdsem_bend180: ComponentSpec = "cdsem_bend180",
    text_size: float = 1,
) -> Component:
    """Column with all optical PCMs.

    Args:
        widths: for straight lines.
        dense_lines_width: in um.
        dense_lines_width_difference: in um.
        dense_lines_gap: in um.
        dense_lines_labels: strings.
        straight: spec.
        bend90: spec.
        cross_section: spec.
        text: spec.
        spacing: from group to group.
        cdsem_bend180: spec.
        text_size: in um.
    """
    return cf.cdsem_all(
        widths=widths,
        dense_lines_width=dense_lines_width,
        dense_lines_width_difference=dense_lines_width_difference,
        dense_lines_gap=dense_lines_gap,
        dense_lines_labels=dense_lines_labels,
        straight=straight,
        bend90=bend90,
        cross_section=cross_section,
        text=text,
        spacing=spacing,
        cdsem_bend180=cdsem_bend180,
        text_size=text_size,
    )
