from numpy import pi
import pp
from pp.components import taper
from pp.components import bend_circular
from pp.components import waveguide
from pp.routing.manhattan import round_corners
from pp.config import WG_EXPANDED_WIDTH, TAPER_LENGTH
from pp.component import Component
from typing import Callable


@pp.autoname
def delay_snake(
    total_length: float = 160000.0,
    L0: float = 2350.0,
    n: int = 5,
    taper: Callable = taper,
    bend_factory: Callable = bend_circular,
    bend_radius: float = 10.0,
    straight_factory: Callable = waveguide,
    wg_width: float = 0.5,
) -> Component:
    """ Snake input facing west
    Snake output facing east

    Args:
        total_length:
        L0:
        n: number of loops
        taper:
        bend_factory
        bend_radius
        straight_factory
        wg_width

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
      pp.plotgds(c)

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
    for i in range(n):
        y -= 2 * R + epsilon
        path += [(L2, y), (-L0, y)]
        y -= 2 * R + epsilon
        path += [(-L0, y), (L2, y)]

    path = [(round(_x, 3), round(_y, 3)) for _x, _y in path]

    component = pp.Component()
    if taper:
        if callable(taper):
            _taper = taper(
                width1=wg_width, width2=WG_EXPANDED_WIDTH, length=TAPER_LENGTH
            )
        else:
            _taper = taper
    snake = round_corners(path, bend90, straight_factory, taper=_taper)
    component.add(snake)
    component.ports = snake.ports

    pp.port.auto_rename_ports(component)
    return component


if __name__ == "__main__":
    c = delay_snake()
    pp.show(c)
