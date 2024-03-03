from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
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
        kwargs: cross_section settings.
    """
    c = Component()
    grating_coupler = gf.get_component(grating_coupler)

    for i in range(n):
        gc = c << grating_coupler
        gc.rotate(rotation)
        gc.x = i * pitch
        port_name_new = f"o{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    if with_loopback:
        if rotation != 90:
            raise ValueError(
                "with_loopback is currently only programmed to work with rotation = 90"
            )
        routing_xs = gf.get_cross_section(cross_section, **kwargs)
        radius = routing_xs.radius

        steps = (
            {"dy": -radius},
            {"dx": -gc.xsize / 2 - radius},
            {"dy": gc.ysize + 2 * radius},
            {"dx": c.xsize + 2 * radius},
            {"dy": -gc.ysize - 2 * radius},
            {"dx": -gc.xsize / 2 - radius},
        )

        route = gf.routing.get_route_from_steps(
            port1=c.ports["o0"],
            port2=c.ports[f"o{n-1}"],
            steps=steps,
            cross_section=routing_xs,
        )
        c.add(route.references)
        c.ports.pop("o0")
        c.ports.pop(f"o{n-1}")

    return c


if __name__ == "__main__":
    # c = grating_coupler_array()
    c = grating_coupler_array(rotation=90, with_loopback=True, radius=20)
    c.show(show_ports=True)
