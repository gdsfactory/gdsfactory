from typing import Dict, List, Optional

import numpy as np

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.get_bundle_from_waypoints import get_bundle_from_waypoints
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.types import ComponentOrFactory, CrossSectionFactory, Route


def get_bundle_from_steps(
    ports1: List[Port],
    ports2: List[Port],
    steps: Optional[List[Dict[str, float]]] = None,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentOrFactory = straight_function,
    taper_factory: Optional[ComponentOrFactory] = taper_function,
    cross_section: CrossSectionFactory = strip,
    sort_ports: bool = True,
    **kwargs
) -> List[Route]:
    """Returns a list of routes formed by the given waypoints steps
    bends instead of corners and optionally tapers in straight sections.
    Tapering to wider straights reduces the optical loss and phase errors.
    `get_bundle_from_steps` is a manual version of `get_bundle`
    and a more convenient version of `get_bundle_from_waypoints`

    Args:
        port1: start ports (list or dict)
        port2: end ports (list or dict)
        steps: changes that define the route [{'dx': 5}, {'dy': 10}]
        bend: function that returns bends
        straight: function that returns straight waveguides
        taper_factory: function that returns tapers
        cross_section: for routes
        sort_ports: if True sort ports
        kwargs: cross_section settings

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("get_route_from_steps_sample")
        w = gf.components.array(
            gf.partial(gf.c.straight, layer=(2, 0)),
            rows=3,
            columns=1,
            spacing=(0, 50),
        )

        left = c << w
        right = c << w
        right.move((200, 100))
        p1 = left.get_ports_list(orientation=0)
        p2 = right.get_ports_list(orientation=180)

        routes = get_bundle_from_steps(
            p1,
            p2,
            steps=[{"x": 150}],
        )

        for route in routes:
            c.add(route.references)
        c.plot()
        c.show()

    """
    if isinstance(ports1, Port):
        ports1 = [ports1]

    if isinstance(ports2, Port):
        ports2 = [ports2]

    # convert ports dict to list
    if isinstance(ports1, dict):
        ports1 = list(ports1.values())

    if isinstance(ports2, dict):
        ports2 = list(ports2.values())

    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    waypoints = []
    steps = steps or []

    x, y = ports1[0].midpoint
    for d in steps:
        x = d["x"] if "x" in d else x
        x += d.get("dx", 0)
        y = d["y"] if "y" in d else y
        y += d.get("dy", 0)
        waypoints += [(x, y)]

    port2 = ports2[0]
    x2, y2 = port2.midpoint
    orientation = int(port2.orientation)

    if orientation in [0, 180]:
        waypoints += [(x, y2)]
    elif orientation in [90, 270]:
        waypoints += [(x2, y)]

    x = cross_section(**kwargs)
    auto_widen = x.info.get("auto_widen", False)
    width1 = x.info.get("width")
    width2 = x.info.get("width_wide") if auto_widen else width1
    taper_length = x.info.get("taper_length")
    waypoints = np.array(waypoints)

    if auto_widen:
        taper = (
            taper_factory(
                length=taper_length,
                width1=width1,
                width2=width2,
                cross_section=cross_section,
                **kwargs,
            )
            if callable(taper_factory)
            else taper_factory
        )
    else:
        taper = None

    return get_bundle_from_waypoints(
        ports1=ports1,
        ports2=ports2,
        waypoints=waypoints,
        bend=bend,
        straight=straight,
        taper=taper,
        cross_section=cross_section,
        **kwargs,
    )


get_bundle_from_steps_electrical = gf.partial(
    get_bundle_from_steps, bend=wire_corner, cross_section=gf.cross_section.metal3
)


if __name__ == "__main__":
    # c = test_route_from_steps()
    c = gf.Component("get_route_from_steps_sample")

    w = gf.components.array(
        gf.partial(gf.c.straight, layer=(2, 0)),
        rows=3,
        columns=1,
        spacing=(0, 50),
    )

    left = c << w
    right = c << w
    right.move((200, 100))
    p1 = left.get_ports_list(orientation=0)
    p2 = right.get_ports_list(orientation=180)

    routes = get_bundle_from_steps_electrical(
        p1,
        p2,
        steps=[{"x": 150}],
    )

    # routes = get_bundle_from_waypoints(
    #     p1,
    #     p2,
    #     # waypoints=[(150,0), (150, 100)],
    #     waypoints=[(150, 0)],
    # )

    for route in routes:
        c.add(route.references)

    # route = gf.routing.get_route_from_steps(
    #     p1[-1],
    #     p2[0],
    #     steps=[{"x": 100}, {"y": 100}],
    # )
    # c.add(route.references)
    c.show()
