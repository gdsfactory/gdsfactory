from __future__ import annotations

from functools import partial

import numpy as np
from kfactory.routing.optical import OpticalManhattanRoute

import gdsfactory as gf
from gdsfactory.port import Port
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    Component,
    ComponentSpec,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
)


def route_single_from_steps(
    component: Component,
    port1: Port,
    port2: Port,
    steps: list[dict[str, float]] | None = None,
    bend: ComponentSpec = "bend_euler",
    taper: ComponentSpec | None = "taper",
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = "strip",
    auto_widen: bool = False,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    **kwargs,
) -> OpticalManhattanRoute:
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
        taper: taper spec.
        cross_section: cross_section spec.
        auto_widen: if True, tapers to wider straights.
        port_type: optical or electrical.
        allow_width_mismatch: if True, allows width mismatch.
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
    x, y = port1.dcenter

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

    port_type = port_type or port1.port_type
    return route_single(
        component=component,
        port1=port1,
        port2=port2,
        waypoints=waypoints,
        bend=bend,
        taper=taper,
        cross_section=cross_section,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        **kwargs,
    )


route_single_from_steps_electrical = partial(
    route_single_from_steps, bend="wire_corner", taper=None, cross_section="metal3"
)


if __name__ == "__main__":
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
