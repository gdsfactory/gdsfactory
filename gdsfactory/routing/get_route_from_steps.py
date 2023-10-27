from __future__ import annotations

from functools import partial

import numpy as np
from kfactory.routing.optical import OpticalManhattanRoute

import gdsfactory as gf
from gdsfactory.components.via_corner import via_corner
from gdsfactory.port import Port
from gdsfactory.routing.get_route import place_route
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    Component,
    ComponentSpec,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
)


def get_route_from_steps(**kwargs) -> None:
    raise ValueError(
        "get_route is not supported in gdsfactory >=8. Use place_route instead!"
    )


def place_route_from_steps(
    component: Component,
    port1: Port,
    port2: Port,
    steps: list[dict[str, float]] | None = None,
    bend: ComponentSpec = "bend_euler",
    taper: ComponentSpec | None = "taper",
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "xs_sc",
    **kwargs,
) -> OpticalManhattanRoute:
    """Places a route formed by the given waypoints steps.

    Uses smooth euler bends instead of corners and tapers in straight sections.
    Tapering to wider straights reduces the optical loss when auto_widen=True.
    `get_route_from_steps` is a manual version of `get_route`
    and a more concise and convenient version of `get_route_from_waypoints`

    Args:
        port1: start port.
        port2: end port.
        steps: that define the route (x, y, dx, dy) [{'dx': 5}, {'dy': 10}].
        bend: function that returns bends.
        straight: straight spec.
        taper: taper spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("get_route_from_steps_sample")
        w = gf.components.straight()
        left = c << w
        right = c << w
        right.d.move((100, 80))

        obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
        obstacle1 = c << obstacle
        obstacle2 = c << obstacle
        obstacle1.d.ymin = 40
        obstacle2.d.xmin = 25

        p1 = left.ports['o2']
        p2 = right.ports['o2']
        gf.routing.place_route_from_steps(
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
    x, y = port1.d.center
    x2, y2 = port2.d.center

    waypoints = []
    steps = steps or []

    for d in steps:
        if not STEP_DIRECTIVES.issuperset(d):
            invalid_step_directives = list(set(d.keys()) - STEP_DIRECTIVES)
            raise ValueError(
                f"Invalid step directives: {invalid_step_directives}."
                f"Valid directives are {list(STEP_DIRECTIVES)}"
            )
        x = d["x"] if "x" in d else x
        x += d.get("dx", 0)
        y = d["y"] if "y" in d else y
        y += d.get("dy", 0)
        waypoints += [(x, y)]

    waypoints = np.array(waypoints)

    if isinstance(cross_section, list | tuple):
        xs_list = []
        for element in cross_section:
            xs, angles = element
            xs = gf.get_cross_section(xs)
            xs = xs.copy(**kwargs)  # Shallow copy
            xs_list.append((xs, angles))
        cross_section = xs_list

    else:
        cross_section = gf.get_cross_section(cross_section)
        x = cross_section = cross_section.copy(**kwargs)
        auto_widen = x.auto_widen

        if auto_widen:
            taper = gf.get_component(
                taper,
                length=x.taper_length,
                width1=x.width,
                width2=x.width_wide,
                cross_section=cross_section,
                **kwargs,
            )
        else:
            taper = None

    return place_route(
        component=component,
        port1=port1,
        port2=port2,
        waypoints=waypoints,
        bend=bend,
        taper=taper,
        cross_section=cross_section,
        **kwargs,
    )


get_route_from_steps_electrical = partial(
    get_route_from_steps, bend="wire_corner", taper=None, cross_section="xs_m3"
)

get_route_from_steps_electrical_multilayer = partial(
    get_route_from_steps,
    bend=via_corner,
    taper=None,
    cross_section=[
        (gf.cross_section.metal2, (90, 270)),
        ("xs_metal_routing", (0, 180)),
    ],
)


def test_route_from_steps():
    c = gf.Component()
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.d.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.d.ymin = 40
    obstacle2.d.xmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    place_route_from_steps(
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


if __name__ == "__main__":
    test_route_from_steps()

    import gdsfactory as gf

    c = gf.Component("get_route_from_steps_sample")
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.d.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.d.ymin = 40
    obstacle2.d.xmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    place_route_from_steps(
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
    # pt.move((100, 200))
    # route = gf.routing.get_route_from_steps_electrical(
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
