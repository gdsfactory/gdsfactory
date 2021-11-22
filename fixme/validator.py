"""FIXME
instead of type annotations we can create pydantic base models for all Components
this will give more readable error messages
"""

import gdsfactory as gf
from gdsfactory.types import ComponentFactory


@gf.cell
def straight_with_bend(
    straight: ComponentFactory = gf.c.straight, bend: ComponentFactory = gf.c.bend_euler
):
    c = gf.Component()
    s = c << straight()
    b = c << bend()
    b.connect("o1", s.ports["o2"])
    return c


if __name__ == "__main__":

    c = straight_with_bend(straight=3)
    c.show()
