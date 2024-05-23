from __future__ import annotations

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical import grating_coupler_elliptical
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def grating_coupler_array(
    grating_coupler: ComponentSpec = grating_coupler_elliptical,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = -90,
    with_loopback: bool = False,
    cross_section: CrossSectionSpec = "strip",
    straight_to_grating_spacing: float = 10.0,
    centered: bool = True,
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
        centered: if True, centers the array around the origin.
    """
    c = Component()
    grating_coupler = gf.get_component(grating_coupler)

    for i in range(n):
        gc = c << grating_coupler
        gc.d.rotate(rotation)
        gc.d.x = i * pitch if not centered else (i - (n - 1) / 2) * pitch
        port_name_new = f"o{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    if with_loopback:
        if rotation != -90:
            raise ValueError(
                "with_loopback is currently only programmed to work with rotation = 90"
            )
        routing_xs = gf.get_cross_section(cross_section)
        radius = routing_xs.radius

        port0 = c.ports["o0"]
        port1 = c.ports[f"o{n-1}"]
        radius = radius
        radius_dbu = round(radius / c.kcl.dbu)

        waypoints = kf.routing.optical.route_loopback(
            port0,
            port1,
            bend90_radius=radius_dbu,
            d_loop=round(straight_to_grating_spacing / c.kcl.dbu)
            + radius_dbu
            + gc.ysize,
        )

        gf.routing.route_single(
            c,
            port1=port0,
            port2=port1,
            waypoints=waypoints,
            cross_section=cross_section,
        )

    return c


if __name__ == "__main__":
    c = grating_coupler_array(with_loopback=True, centered=True)
    c.show()
