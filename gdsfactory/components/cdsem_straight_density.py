"""CD SEM structures."""
from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Floats, Optional

text_rectangular_mini = partial(text_rectangular, size=1)

widths = 10 * [0.3]
gaps = 10 * [0.3]


@cell
def cdsem_straight_density(
    widths: Floats = widths,
    gaps: Floats = gaps,
    length: float = 420.0,
    label: str = "",
    cross_section: CrossSectionFactory = strip,
    text: Optional[ComponentFactory] = text_rectangular_mini,
) -> Component:
    """Returns sweep of dense straight lines

    Args:
        widths: list of widths
        gaps: list of gaps
        length: of the lines
        label: defaults to widths[0] gaps[0]
        cross_section:
        text: optional function for text
    """
    c = Component()
    label = label or f"{int(widths[0]*1e3)} {int(gaps[0]*1e3)}"

    ymin = 0
    for width, gap in zip(widths, gaps):
        cross_section = partial(cross_section, width=width)
        tooth_ref = c << straight_function(length=length, cross_section=cross_section)
        tooth_ref.ymin = ymin
        ymin += width + gap

    if text:
        marker_label = c << text(text=f"{label}")
        marker_label.xmin = tooth_ref.xmax + 5
    return c


if __name__ == "__main__":
    c = cdsem_straight_density()
    c.show()
