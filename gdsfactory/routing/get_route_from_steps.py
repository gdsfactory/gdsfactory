from typing import Dict, List, Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import StrOrDict, get_cross_section
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.types import ComponentOrFactory, Route


def get_route_from_steps(
    port1: Port,
    port2: Port,
    steps: List[Dict[str, float]],
    bend_factory: ComponentOrFactory = bend_euler,
    straight_factory: ComponentOrFactory = straight,
    taper_factory: Optional[ComponentOrFactory] = taper_function,
    waveguide: StrOrDict = "strip",
    **waveguide_settings,
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
        bend_factory: function that returns bends
        straight_factory: function that returns straight waveguides
        taper_factory: function that returns tapers
        waveguide_settings

    .. plot::
        :include-source:

        import gdsfactory as gf

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

        p1 = left.ports["E0"]
        p2 = right.ports["E0"]
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
        c.show()

    """
    x, y = port1.midpoint
    x2, y2 = port2.midpoint

    waypoints = [(x, y)]

    for d in steps:
        x = d["x"] if "x" in d else x
        x += d.get("dx", 0)
        y = d["y"] if "y" in d else y
        y += d.get("dy", 0)
        waypoints += [(x, y)]

    waypoints += [(x2, y2)]

    x = get_cross_section(waveguide, **waveguide_settings)
    waveguide_settings = x.info
    auto_widen = waveguide_settings.get("auto_widen", False)
    width1 = waveguide_settings.get("width")
    width2 = waveguide_settings.get("width_wide") if auto_widen else width1
    taper_length = waveguide_settings.get("taper_length")
    waypoints = np.array(waypoints)

    if auto_widen:
        taper = (
            taper_factory(
                length=taper_length,
                width1=width1,
                width2=width2,
                waveguide=waveguide,
                **waveguide_settings,
            )
            if callable(taper_factory)
            else taper_factory
        )
    else:
        taper = None

    return round_corners(
        points=waypoints,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        taper=taper,
        waveguide=waveguide,
        **waveguide_settings,
    )


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

    p1 = left.ports["E0"]
    p2 = right.ports["E0"]

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

    assert route.length == 187.196
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
    assert route.length == 187.196
    return c


if __name__ == "__main__":
    c = test_route_from_steps()
    c.show()
