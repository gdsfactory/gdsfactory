from typing import Dict, List, Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentOrFactory, CrossSectionFactory, Route


def get_route_from_steps(
    port1: Port,
    port2: Port,
    steps: Optional[List[Dict[str, float]]] = None,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentOrFactory = straight_function,
    taper: Optional[ComponentOrFactory] = taper_function,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Route:
    """Returns a route formed by the given waypoints steps
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss.
    `get_route_from_steps` is a manual version of `get_route`
    and a more concise and convenient version of `get_route_from_waypoints`

    Args:
        port1: start port
        port2: end port
        steps: changes that define the route [{'dx': 5}, {'dy': 10}]
        bend: function that returns bends
        straight: function that returns straight waveguides
        taper: function that returns tapers
        cross_section
        **kwargs: cross_section settings

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("get_route_from_steps_sample")
        w = gf.components.straight()
        left = c << w
        right = c << w
        right.move((100, 80))

        obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
        obstacle1 = c << obstacle
        obstacle2 = c << obstacle
        obstacle1.ymin = 40
        obstacle2.xmin = 25

        p1 = left.ports['o2']
        p2 = right.ports['o2']
        route = gf.routing.get_route_from_steps(
            port1=p1,
            port2=p2,
            steps=[
                {"x": 20},
                {"y": 20},
                {"x": 120},
                {"y": 80},
            ],
        )
        c.add(route.references)
        c.plot()
        c.show()

    """
    x, y = port1.midpoint
    x2, y2 = port2.midpoint

    waypoints = [(x, y)]
    steps = steps or []

    for d in steps:
        x = d["x"] if "x" in d else x
        x += d.get("dx", 0)
        y = d["y"] if "y" in d else y
        y += d.get("dy", 0)
        waypoints += [(x, y)]

    waypoints += [(x2, y2)]

    x = cross_section(**kwargs)
    auto_widen = x.info.get("auto_widen", False)
    width1 = x.info.get("width")
    width2 = x.info.get("width_wide") if auto_widen else width1
    taper_length = x.info.get("taper_length")
    waypoints = np.array(waypoints)

    if auto_widen:
        taper = (
            taper(
                length=taper_length,
                width1=width1,
                width2=width2,
                cross_section=cross_section,
                **kwargs,
            )
            if callable(taper)
            else taper
        )
    else:
        taper = None

    return round_corners(
        points=waypoints,
        bend=bend,
        straight=straight,
        taper=taper,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def test_route_from_steps() -> gf.Component:
    c = gf.Component("get_route_from_steps_sample")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10))
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.ymin = 40
    obstacle2.xmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]

    route = get_route_from_steps(
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20, "y": 0},
            {"x": 20, "y": 20},
            {"x": 120, "y": 20},
            {"x": 120, "y": 80},
        ],
    )

    length = 186.548

    assert route.length == length, route.length
    route = gf.routing.get_route_from_steps(
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20},
            {"y": 20},
            {"x": 120},
            {"y": 80},
        ],
        layer=(2, 0),
    )
    c.add(route.references)
    assert route.length == length, route.length
    return c


if __name__ == "__main__":
    # c = test_route_from_steps()
    c = gf.Component("get_route_from_steps_sample")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.move((100, 80))

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]

    route = get_route_from_steps(
        port1=p2,
        port2=p1,
        # steps=[
        #     {"x": 20, "y": 0},
        #     {"x": 20, "y": 20},
        #     {"x": 120, "y": 20},
        #     {"x": 120, "y": 80},
        # ],
    )
    c.add(route.references)
    c.show()
