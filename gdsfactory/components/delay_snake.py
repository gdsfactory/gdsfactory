from numpy import pi

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentFactory


@gf.cell
def delay_snake(
    wg_width: float = 0.5,
    wg_width_wide: float = 2.0,
    total_length: float = 1600.0,
    L0: float = 5.0,
    taper_length: float = 10.0,
    n: int = 2,
    taper: ComponentFactory = taper_function,
    bend_factory: ComponentFactory = bend_euler,
    straight_factory: ComponentFactory = straight_function,
    **kwargs
) -> Component:
    """Snake input facing west
    Snake output facing east

    Args:
        wg_width
        total_length:
        L0: initial offset
        n: number of loops
        taper: taper library
        bend_factory
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
    bend90 = bend_factory(width=wg_width, **kwargs)
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

    component = gf.Component()
    if taper:
        _taper = taper(
            width1=wg_width, width2=wg_width_wide, length=taper_length, **kwargs
        )
    route = round_corners(
        points=path,
        bend_factory=bend90,
        straight_factory=straight_factory,
        taper=_taper,
        width_wide=wg_width_wide,
        **kwargs
    )
    component.add(route.references)
    component.add_port("o1", port=route.ports[0])
    component.add_port("o2", port=route.ports[1])
    return component


if __name__ == "__main__":
    c = delay_snake(layer=(2, 0), auto_widen=False)
    c.show(show_ports=True)
