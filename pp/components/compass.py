from typing import Tuple

import pp
from pp.component import Component


@pp.cell
def compass(
    size: Tuple[float, float] = (4, 2), layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """ Rectangular contact pad with centered ports on rectangle edges
    (north, south, east, and west)

    Args:
        size: tuple (4, 2)
        layer: tuple (int, int)

    .. plot::
      :include-source:

      import pp

      c = pp.c.compass(size=(4, 2), layer=pp.LAYER.WG)
      pp.plotgds(c)
    """

    c = pp.Component()
    r = pp.c.rectangle(size=size, layer=layer)
    dx = size[0]
    dy = size[1]

    rr = r.ref(position=(-dx / 2, -dy / 2))
    c.add(rr)
    c.add_port(name="N", midpoint=[0, dy / 2], width=dx, orientation=90, layer=layer)
    c.add_port(name="S", midpoint=[0, -dy / 2], width=dx, orientation=-90, layer=layer)
    c.add_port(name="E", midpoint=[dx / 2, 0], width=dy, orientation=0, layer=layer)
    c.add_port(name="W", midpoint=[-dx / 2, 0], width=dy, orientation=180, layer=layer)

    return c


if __name__ == "__main__":
    c = compass(size=(1, 2), layer=pp.LAYER.WG)
    pp.show(c)
