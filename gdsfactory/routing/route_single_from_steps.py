from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Literal

from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory._deprecation import deprecate
from gdsfactory.component import Component
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinate,
    CrossSectionSpec,
    Port,
)


def route_single_from_steps(
    component: Component,
    port1: Port,
    port2: Port,
    steps: Sequence[Mapping[Literal["x", "y", "dx", "dy"], int | float]] | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    auto_taper: bool = True,
    **kwargs: Any,
) -> ManhattanRoute:
    """Places a route formed by the given waypoints steps.

    Uses smooth euler bends instead of corners and tapers in straight sections.
    Tapering to wider straights reduces the optical loss when auto_widen=True.
    `route_single_from_steps` is a manual version of `route_single`
    and a more concise and convenient version of `route_single_from_waypoints`

    Args:
        component: to add the route to.
        port1: start port.
        port2: end port.
        steps: that define the route (x, y, dx, dy) [{'dx': 5}, {'dy': 10}].
        bend: function that returns bends.
        straight: straight spec.
        cross_section: cross_section spec.
        port_type: optical or electrical.
        allow_width_mismatch: if True, allows width mismatch.
        auto_taper: if True, adds taper to the route.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("route_single_from_steps_sample")
        w = gf.components.straight()
        left = c << w
        right = c << w
        right.dmove((100, 80))

        obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
        obstacle1 = c << obstacle
        obstacle2 = c << obstacle
        obstacle1.dymin = 40
        obstacle2.dxmin = 25

        p1 = left.ports['o2']
        p2 = right.ports['o2']
        gf.routing.route_single_from_steps(
            c,
            port1=p1,
            port2=p2,
            steps=[
                {"x": 20},
                {"y": 20},
                {"x": 120},
                {"y": 80},
            ],
        )
        c.plot()

    """
    deprecate("route_single_from_steps", "route_single")
    x, y = port1.dcenter
    waypoints: list[Coordinate] = []
    steps = list(steps or [])

    for d in steps:
        if not STEP_DIRECTIVES.issuperset(d):
            invalid_step_directives = list(set(map(str, d.keys())) - STEP_DIRECTIVES)
            raise ValueError(
                f"Invalid step directives: {invalid_step_directives}."
                f"Valid directives are {list(STEP_DIRECTIVES)}"
            )
        x = d.get("x", x) + d.get("dx", 0)
        y = d.get("y", y) + d.get("dy", 0)
        waypoints += [(x, y)]

    port_type = port_type or port1.port_type
    return route_single(
        component=component,
        port1=port1,
        port2=port2,
        waypoints=waypoints,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        auto_taper=auto_taper,
        **kwargs,
    )


route_single_from_steps_electrical = partial(
    route_single_from_steps,
    bend="wire_corner",
    cross_section="metal3",  # taper=None,
)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("route_single_from_steps_sample")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.dmove((500, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.dymin = 40
    obstacle2.dxmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    route_single_from_steps(
        c,
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20},
            {"y": 20},
            {"x": 120},
            {"y": 80},
        ],
    )
    c.show()

    # c = gf.Component("pads_route_from_steps")
    # pt = c << gf.components.pad_array(orientation=270, columns=3)
    # pb = c << gf.components.pad_array(orientation=90, columns=3)
    # pt.dmove((100, 200))
    # route = gf.routing.route_single_from_steps_electrical(
    #     pb.ports["e11"],
    #     pt.ports["e11"],
    #     steps=[
    #         {"y": 200},
    #         # {"z": 200},
    #     ],
    #     # cross_section='metal_routing',
    #     # bend=gf.components.wire_corner,
    # )
    # c.add(route.references)
    # c.show()
