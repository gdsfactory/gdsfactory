from numpy import pi

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def delay_snake(
    total_length: float = 1600.0,
    L0: float = 5.0,
    n: int = 2,
    bend: ComponentSpec = "bend_euler",
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Snake input facing west output facing east.

    Args:
        total_length: of the delay.
        L0: initial xoffset.
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

    c = gf.Component()
    route = round_corners(
        points=path, bend=bend90, cross_section=cross_section, **kwargs
    )
    c.add(route.references)
    c.add_port("o1", port=route.ports[0])
    c.add_port("o2", port=route.ports[1])
    return c


if __name__ == "__main__":
    c = delay_snake(cross_section="strip_auto_widen", auto_widen_minimum_length=50)
    c.show(show_ports=True)
