from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def delay_snake(
    length: float = 1600.0,
    L0: float = 5.0,
    n: int = 2,
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns Snake with a starting straight and 90 bends.

    Input faces west output faces east.

    Args:
        length: delay length in um.
        L0: initial xoffset in um.
        n: number of loops.
        bend: bend spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

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
    bend90 = gf.get_component(bend, cross_section=cross_section, **kwargs)
    dy = bend90.info["dy"]
    DL = (length + L0 - n * (np.pi * dy + epsilon)) / (2 * n + 1)
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

    c = gf.Component()
    route = round_corners(
        points=path, bend=bend90, cross_section=cross_section, **kwargs
    )
    c.add(route.references)
    c.add_port("o1", port=route.ports[0])
    c.add_port("o2", port=route.ports[1])
    return c


def test_delay_snake_length():
    length = 200.0
    c = delay_snake(n=1, length=length, cross_section="strip_no_pins")
    length_computed = c.area() / 0.5
    np.isclose(length, length_computed)
    return c


if __name__ == "__main__":
    c = test_delay_snake_length()
    # c = delay_snake(cross_section="strip_auto_widen", auto_widen_minimum_length=50)
    c.show(show_ports=True)
