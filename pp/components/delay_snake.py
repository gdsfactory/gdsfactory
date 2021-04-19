from numpy import pi

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.straight import straight as straight_function
from pp.components.taper import taper as taper_function
from pp.config import TAPER_LENGTH, WG_EXPANDED_WIDTH
from pp.routing.manhattan import round_corners
from pp.types import ComponentFactory


@cell
def delay_snake(
    wg_width: float = 0.5,
    total_length: float = 1600.0,
    L0: float = 5.0,
    n: int = 2,
    taper: ComponentFactory = taper_function,
    bend_factory: ComponentFactory = bend_euler,
    bend_radius: float = 10.0,
    straight_factory: ComponentFactory = straight_function,
) -> Component:
    """Snake input facing west
    Snake output facing east

    Args:
        wg_width
        total_length:
        L0: initial offset
        n: number of loops
        taper: taper factory
        bend_factory
        bend_radius
        straight_factory

    .. code::

       | L0 |    L2        |

            ->-------------|
                           | pi * radius
       |-------------------|
       |
       |------------------->

       |        DL         |


    """
    epsilon = 0.1
    bend90 = bend_factory(radius=bend_radius, width=wg_width)
    dy = bend90.dy
    DL = (total_length + L0 - n * (pi * dy + epsilon)) / (2 * n + 1)
    L2 = DL - L0
    assert (
        L2 > 0
    ), "Snake is too short: either reduce L0, increase the total length,\
    or decrease n"

    y = 0
    path = [(0, y), (L2, y)]
    for _i in range(n):
        y -= 2 * dy + epsilon
        path += [(L2, y), (-L0, y)]
        y -= 2 * dy + epsilon
        path += [(-L0, y), (L2, y)]

    path = [(round(_x, 3), round(_y, 3)) for _x, _y in path]

    component = pp.Component()
    if taper:
        _taper = taper(width1=wg_width, width2=WG_EXPANDED_WIDTH, length=TAPER_LENGTH)
    route_snake = round_corners(
        points=path,
        bend_factory=bend90,
        straight_factory=straight_factory,
        taper=_taper,
    )
    component.add(route_snake["references"])
    component.ports = route_snake["ports"]

    pp.port.auto_rename_ports(component)
    return component


if __name__ == "__main__":
    c = delay_snake()
    c.show()
