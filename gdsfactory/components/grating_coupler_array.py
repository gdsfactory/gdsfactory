from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler_elliptical import grating_coupler_elliptical
from gdsfactory.typings import ComponentSpec


@gf.cell
def grating_coupler_array(
    grating_coupler: ComponentSpec = grating_coupler_elliptical,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = 0,
    with_loopback: bool = False,
    cross_section: str = "xs_sc",
    bend: ComponentSpec = bend_euler,
    grating_coupler_spacing: float = 0.0,
    **kwargs,
) -> Component:
    """Array of grating couplers.

    Args:
        grating_coupler: ComponentSpec.
        pitch: x spacing.
        n: number of grating couplers.
        port_name: port name.
        rotation: rotation angle for each reference.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        cross_section: cross_section for the routing.
        bend: bend component.
        grating_coupler_spacing: spacing between grating couplers.
        kwargs: cross_section settings.
    """
    c = Component()
    grating_coupler = gf.get_component(grating_coupler)

    for i in range(n):
        gc = c << grating_coupler
        gc.rotate(rotation)
        gc.x = i * pitch
        port_name_new = f"o{i+1}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    if with_loopback:
        if rotation != 90:
            raise ValueError(
                "with_loopback is currently only programmed to work with rotation = 90"
            )
        routing_xs = gf.get_cross_section(cross_section, **kwargs)
        radius = routing_xs.radius
        bend = bend(radius=radius, angle=180, cross_section=routing_xs)
        sw = gf.c.straight(
            cross_section=routing_xs, length=max(gc.size) + grating_coupler_spacing
        )

        b1 = c << bend
        b2 = c << bend
        b1.mirror()
        b1.connect("o1", c.ports["o1"])
        b2.connect("o1", c.ports[f"o{n}"])

        s1 = c << sw
        s2 = c << sw
        s1.connect("o1", b1.ports["o2"])
        s2.connect("o1", b2.ports["o2"])

        route = gf.routing.get_route(
            s1.ports["o2"],
            s2.ports["o2"],
            cross_section=routing_xs,
        )
        c.add(route.references)
        c.ports.pop("o1")
        c.ports.pop(f"o{n}")

    return c


if __name__ == "__main__":
    # c = grating_coupler_array()
    c = grating_coupler_array(rotation=90, with_loopback=True, radius=50)
    c.show(show_ports=True)
