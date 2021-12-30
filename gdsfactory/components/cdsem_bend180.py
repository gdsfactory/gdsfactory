"""CD SEM structures."""
from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory

LINE_LENGTH = 420.0

text_rectangular_mini = partial(text_rectangular, size=1)


@cell
def cdsem_bend180(
    width: float = 0.5,
    radius: float = 10.0,
    wg_length: float = LINE_LENGTH,
    straight: ComponentFactory = straight_function,
    bend90: ComponentFactory = bend_circular,
    cross_section: CrossSectionFactory = strip,
    text: ComponentFactory = text_rectangular_mini,
) -> Component:
    """

    Args:
        width: of the line
        cladding_offset:
        radius: bend radius
        wg_length

    """
    c = Component()
    r = radius

    cross_section = partial(cross_section, width=width)
    if wg_length is None:
        wg_length = 2 * r

    bend90 = bend90(cross_section=cross_section, radius=r)
    wg = straight(
        cross_section=cross_section,
        length=wg_length,
    )

    # Add the U-turn on straight layer
    b1 = c.add_ref(bend90)
    b2 = c.add_ref(bend90)
    b2.connect("o2", b1.ports["o1"])

    wg1 = c.add_ref(wg)
    wg1.connect("o1", b1.ports["o2"])

    wg2 = c.add_ref(wg)
    wg2.connect("o1", b2.ports["o1"])

    label = c << text(text=str(int(width * 1e3)))
    label.ymax = b2.ymin - 5
    label.x = 0
    b1.rotate(90)
    b2.rotate(90)
    wg1.rotate(90)
    wg2.rotate(90)
    label.rotate(90)
    return c


if __name__ == "__main__":
    c = cdsem_bend180(width=2)
    c.show()
