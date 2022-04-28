from typing import Dict, List, Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Route


def get_route_from_steps(
    port1: Port,
    port2: Port,
    steps: Optional[List[Dict[str, float]]] = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    taper: Optional[ComponentSpec] = "taper",
    cross_section: CrossSectionSpec = strip,
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

    x = gf.get_cross_section(cross_section, **kwargs)
    auto_widen = x.auto_widen
    width1 = x.width
    width2 = x.width_wide if auto_widen else width1
    taper_length = x.taper_length
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


get_route_from_steps_electrical = gf.partial(
    get_route_from_steps, bend="wire_corner", taper=None, cross_section="metal3"
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
    # c = gf.Component("get_route_from_steps_sample")
    # w = gf.components.straight()
    # left = c << w
    # right = c << w
    # right.move((100, 80))

    # p1 = left.ports["o2"]
    # p2 = right.ports["o2"]

    # route = get_route_from_steps(
    #     port1=p2,
    #     port2=p1,
    #     steps=[
    #         {"x": 20, "y": 0},
    #         {"x": 20, "y": 20},
    #         {"x": 120, "y": 20},
    #         {"x": 120, "y": 80},
    #     ],
    # )
    # c.add(route.references)
    # c.show()

    c = gf.Component("pads_route_from_steps")
    pt = c << gf.components.pad_array(orientation=270, columns=3)
    pb = c << gf.components.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps_electrical(
        pb.ports["e11"],
        pt.ports["e11"],
        steps=[
            {"y": 200},
        ],
        # cross_section=gf.cross_section.metal3,
        # bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.show()
