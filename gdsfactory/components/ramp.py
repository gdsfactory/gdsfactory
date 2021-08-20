from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import Layer


@gf.cell
def ramp(
    length: float = 10.0,
    width1: float = 5.0,
    width2: Optional[float] = 8.0,
    layer: Layer = (1, 0),
) -> Component:
    """Return a ramp component. Based on phidl.

    Args:
        length: Length of the ramp section.
        width1: Width of the start of the ramp section.
        width2: Width of the end of the ramp section (defaults to width1).
        layer: Specific layer to put polygon geometry on.

    """
    if width2 is None:
        width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1, width2, 0, 0]
    c = Component()
    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(name="o1", midpoint=[0, width1 / 2], width=width1, orientation=180)
    c.add_port(name="o2", midpoint=[length, width2 / 2], width=width2, orientation=0)
    return c


if __name__ == "__main__":
    c = ramp()
    c.show()
