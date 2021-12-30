"""CD SEM structures."""
from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@cell
def cdsem_straight_density(
    width: float = 0.3,
    trench_width: float = 0.3,
    x: float = 420.0,
    y: float = 50.0,
    margin: float = 2.0,
    label: str = "",
    straight: ComponentFactory = straight_function,
    cross_section: CrossSectionFactory = strip,
    text: ComponentFactory = text_rectangular,
) -> Component:
    """Returns sweep of dense straight lines

    Args:
        width: width
        trench_width: trench_width
    """
    c = Component()
    period = width + trench_width
    n_o_lines = int((y - 2 * margin) / period)
    length = x - 2 * margin

    cross_section = partial(cross_section, width=width)
    tooth = straight(length=length, cross_section=cross_section)

    for i in range(n_o_lines):
        tooth_ref = c.add_ref(tooth)
        tooth_ref.movey((-n_o_lines / 2 + 0.5 + i) * period)
        c.absorb(tooth_ref)

    marker_label = text(text=f"{label}")
    _marker_label = c.add_ref(marker_label)
    _marker_label.move((length + 3, 10.0))
    c.absorb(_marker_label)
    return c


if __name__ == "__main__":
    c = cdsem_straight_density()
    c.show()
