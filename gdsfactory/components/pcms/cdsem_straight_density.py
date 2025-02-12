"""CD SEM structures."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats

widths = 10 * (0.3,)
gaps = 10 * (0.3,)


@gf.cell
def cdsem_straight_density(
    widths: Floats = widths,
    gaps: Floats = gaps,
    length: float = 420.0,
    label: str = "",
    cross_section: CrossSectionSpec = "strip_no_ports",
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
    c = Component()
    label = label or f"{int(widths[0] * 1e3)} {int(gaps[0] * 1e3)}"

    ymin = 0.0
    tooth_ref: ComponentReference | None = None
    for width, gap in zip(widths, gaps):
        tooth_ref = c << gf.c.straight(
            length=length, cross_section=cross_section, width=width
        )
        tooth_ref.dymin = ymin
        ymin += width + gap

    if text and tooth_ref is not None:
        marker_label = c << gf.get_component(text, text=f"{label}", size=text_size)
        marker_label.dxmin = tooth_ref.dxmax + 5
    return c


if __name__ == "__main__":
    c = cdsem_straight_density()
    c.show()
