"""Add loopback reference for a grating coupler array."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import Component, ComponentSpec


@gf.cell
def add_loopback(
    component: ComponentSpec,
    port1_name: str,
    port2_name: str,
    grating: ComponentSpec,
    grating_separation: float = 127.0,
    grating_rotation: int = -90,
    grating_port_name: str = "o1",
    bend: ComponentSpec = bend_euler,
    south_waveguide_spacing: float | None = None,
    inside: bool = True,
    **kwargs,
) -> Component:
    """Return loopback (grating coupler align reference) references.

    Input grating generated on the left of port1
    Output grating generated on the right of port2

    Args:
        component: input component.
        port1_name: start port.
        port2_name: end port.
        grating: fiber coupler.
        grating_separation: grating pitch.
        grating_rotation: in degrees.
        grating_port_name: fiber port name for grating coupler.
        bend: bend spec.
        south_waveguide_spacing: spacing from loopback to grating_coupler.dymin
        inside: add loopback inside.
        kwargs: cross_section settings.

    .. code::

    inside = True
         ______                     ______
        |<-separation  |     |      |     |
        |      |       |     |      |     |
       GC      |    port1  port2    |    GC___
               |                    |       |
               |                    |       | south_waveguide_spacing
               |____________________|      _|_

    inside = False
                ______                    _______
               |     |                   |      |
               |     |       |     |     |      |
               |     GC   port1  port2   GC     |      ___
               |                                |       |
               |                                |       | south_waveguide_spacing
               |________________________________|      _|_
    """
    port1 = component.ports[port1_name]
    port2 = component.ports[port2_name]

    c = gf.Component()
    gc = gf.get_component(grating)

    y0 = port1.dy
    x0 = port1.dx - grating_separation
    x1 = port2.dx + grating_separation

    gca1 = c << gc
    gca1.drotate(grating_rotation)
    gca1.dmove((x0, y0))

    gca2 = c << gc
    gca2.drotate(grating_rotation)
    gca2.dmove((x1, y0))

    p0 = gca1.ports[grating_port_name].dcenter
    p1 = gca2.ports[grating_port_name].dcenter
    bend90 = bend(**kwargs)

    a = abs(bend90.info["dy"]) if hasattr(bend90, "dx") else bend90.dxsize + 0.5
    b = max(2 * a, grating_separation / 2)
    b = b if inside else -b

    south_waveguide_spacing = (
        south_waveguide_spacing
        if south_waveguide_spacing is not None
        else -gc.dxsize - 5.0
    )

    waypoints = [
        p0 + (0, a),
        p0 + (b, a),
        p0 + (b, south_waveguide_spacing),
        p1 + (-b, south_waveguide_spacing),
        p1 + (-b, a),
        p1 + (0, a),
    ]
    route_single(
        c, port1=port1, port2=port2, waypoints=waypoints, bend=bend90, **kwargs
    )
    return c


if __name__ == "__main__":
    wg = gf.components.straight()
    c = add_loopback(
        wg,
        "o1",
        "o2",
        grating=gf.components.grating_coupler_te,
        inside=False,
    )
    c.show()
