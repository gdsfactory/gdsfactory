from __future__ import annotations

import math

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Port


def route_bundle_sbend(
    component: Component,
    port1: Port,
    port2: Port,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    allow_layer_mismatch: bool = False,
    allow_width_mismatch: bool = False,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
) -> ComponentReference:
    """Returns an Sbend to connect two ports.

    Args:
        component: to add the route to.
        port1: start port.
        port2: end port.
        bend_s: Sbend component.
        cross_section: cross_section.
        allow_layer_mismatch: allow layer mismatch.
        allow_width_mismatch: allow width mismatch.
        start_straight_length: length of the straight at the start of the route.
        end_straight_length: length of the straight at the end of the route.

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.movex(50)
        mmi2.movey(5)
        route = gf.routing.route_bundle_sbend(c, mmi1.ports['o2'], mmi2.ports['o1'])
        c.plot()
        ```
    """
    ysize = port2.center[1] - port1.center[1]
    xsize = port2.center[0] - port1.center[0]

    # We need to act differently if the route is orthogonal in x
    # or orthogonal in y
    size = (xsize, ysize) if port1.orientation in [0, 180] else (ysize, -xsize)
    straight_length = start_straight_length + end_straight_length
    size = (size[0] - math.copysign(straight_length, size[0]), size[1])
    bend = gf.get_component(bend_s, size=size, cross_section=cross_section)

    port_names = [port.name for port in bend.ports]
    if start_straight_length:
        bend = gf.components.extend_ports(
            bend,
            port_names=(port_names[0],),
            extension=gf.components.straight(
                length=start_straight_length,
                cross_section=cross_section,
                width=bend.ports[port_names[0]].width,
            ),
        )
    if end_straight_length:
        bend = gf.components.extend_ports(
            bend,
            port_names=(port_names[1],),
            extension=gf.components.straight(
                length=end_straight_length,
                cross_section=cross_section,
                width=bend.ports[port_names[1]].width,
            ),
        )

    bend_ref = component << bend
    bend_ref.connect(
        bend_ref.ports[0],
        port1,
        allow_layer_mismatch=allow_layer_mismatch,
        allow_width_mismatch=allow_width_mismatch,
    )

    orthogonality_error = abs(abs(port1.orientation - port2.orientation) - 180)
    if orthogonality_error > 0.1:
        raise ValueError(
            f"Ports need to have orthogonal orientation {orthogonality_error}\n"
            f"port1 = {port1.orientation} deg and port2 = {port2.orientation}"
        )
    return bend_ref


# Deprecated alias
route_single_sbend = route_bundle_sbend
