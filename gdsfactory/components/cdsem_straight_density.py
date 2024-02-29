"""CD SEM structures."""
from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.typings import (
    ComponentFactory,
    CrossSectionSpec,
    CrossSectionSpecs,
    Floats,
    Tuple,
)

text_rectangular_mini = partial(text_rectangular, size=1)

widths = 10 * (0.3,)
gaps = 10 * (0.3,)


@cell
def cdsem_straight_density(
    widths: Floats = widths,
    gaps: Floats = gaps,
    length: float = 420.0,
    label: str = "",
    cross_section: CrossSectionSpec | CrossSectionSpecs = "xs_sc",
    text: ComponentFactory | None = text_rectangular_mini,
) -> Component:
    """Returns sweep of dense straight lines.

    Args:
        widths: list of widths.
        gaps: list of gaps.
        length: of the lines.
        label: defaults to widths[0] gaps[0].
        cross_section: spec. Can be a list and then each line has a corresponding cross_section.
        text: optional function for text.
    """
    c = Component()
    label = label or f"{int(widths[0]*1e3)} {int(gaps[0]*1e3)}"

    if isinstance(cross_section, Tuple):
        if len(cross_section) != len(widths):
            raise ValueError(
                "The number of specified cross sections does not correspond to the number of widths"
            )
    else:
        cross_section = [cross_section] * len(widths)

    ymin = 0
    for width, gap, xs in zip(widths, gaps, cross_section):
        tooth_ref = c << straight(length=length, cross_section=xs, width=width)
        tooth_ref.ymin = ymin
        ymin += width + gap

    if text:
        marker_label = c << gf.get_component(text, text=f"{label}")
        marker_label.xmin = tooth_ref.xmax + 5
    return c


if __name__ == "__main__":
    c = cdsem_straight_density(
        widths=(0.2, 0.3), gaps=(0.1, 0.2), cross_section="xs_sc"
    )
    c.show(show_ports=True)
