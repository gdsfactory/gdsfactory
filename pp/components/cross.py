from typing import Tuple
import pp
from pp.component import Component
from pp.name import autoname
from pp.layers import LAYER


@autoname
def cross(
    length: float = 10.0,
    width: float = 3.0,
    layer: Tuple[int, int] = LAYER.WG,
    port_type="optical",
) -> Component:
    """Generates a right-angle cross from two rectangles of specified length and width.

    Args:
        length: float Length of the cross from one end to the other
        width: float Width of the arms of the cross
        layer: int, array-like[2], or set Specific layer(s) to put polygon geometry on

    .. plot::
      :include-source:

      import pp

      c = pp.c.cross(length=10, width=3, layer=(1, 0))
      pp.plotgds(c)

    """

    c = pp.Component()
    R = pp.c.rectangle(size=(width, length), layer=layer)
    r1 = c.add_ref(R).rotate(90)
    r2 = c.add_ref(R)
    r1.center = (0, 0)
    r2.center = (0, 0)
    c.add_port(
        "E0",
        width=width,
        layer=layer,
        port_type=port_type,
        orientation=0,
        midpoint=(+length / 2, 0),
    )
    c.add_port(
        "W0",
        width=width,
        layer=layer,
        port_type=port_type,
        orientation=180,
        midpoint=(-length / 2, 0),
    )
    c.add_port(
        "N0",
        width=width,
        layer=layer,
        port_type=port_type,
        orientation=90,
        midpoint=(0, length / 2),
    )
    c.add_port(
        "S0",
        width=width,
        layer=layer,
        port_type=port_type,
        orientation=270,
        midpoint=(0, -length / 2),
    )
    return c


if __name__ == "__main__":
    c = cross()
    cc = pp.routing.add_fiber_array(c)
    # print(c.ports)
    pp.show(cc)
