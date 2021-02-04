from numpy import pi

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.taper import taper as taper_function
from pp.components.waveguide import waveguide as waveguide_function
from pp.config import TAPER_LENGTH, WG_EXPANDED_WIDTH
from pp.routing.manhattan import round_corners
from pp.types import ComponentFactory


@cell
def delay_snake(
    wg_width: float = 0.5,
    total_length: float = 160000.0,
    L0: float = 2350.0,
    n: int = 5,
    taper: ComponentFactory = taper_function,
    bend_factory: ComponentFactory = bend_circular,
    bend_radius: float = 10.0,
    straight_factory: ComponentFactory = waveguide_function,
) -> Component:
    """ Snake input facing west
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

    .. plot::
      :include-source:

      import pp

      c = pp.c.delay_snake(L0=5, total_length=1600, n=2)
      c.plot()

    """
    epsilon = 0.1
    R = bend_radius
    bend90 = bend_factory(radius=R, width=wg_width)
    DL = (total_length + L0 - n * (pi * R + epsilon)) / (2 * n + 1)
    L2 = DL - L0
    assert (
        L2 > 0
    ), "Snake is too short: either reduce L0, increase the total length,\
    or decrease n"

    y = 0
    path = [(0, y), (L2, y)]
    for _i in range(n):
        y -= 2 * R + epsilon
        path += [(L2, y), (-L0, y)]
        y -= 2 * R + epsilon
        path += [(-L0, y), (L2, y)]

    path = [(round(_x, 3), round(_y, 3)) for _x, _y in path]

    component = pp.Component()
    if taper:
        _taper = taper(width1=wg_width, width2=WG_EXPANDED_WIDTH, length=TAPER_LENGTH)
    route_snake = round_corners(path, bend90, straight_factory, taper=_taper)
    component.add(route_snake["references"])
    component.ports = route_snake["ports"]

    pp.port.auto_rename_ports(component)
    return component


if __name__ == "__main__":
    c = delay_snake()
    c.show()
