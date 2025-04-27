from __future__ import annotations

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def grating_coupler_array(
    grating_coupler: ComponentSpec = "grating_coupler_elliptical",
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = -90,
    with_loopback: bool = False,
    cross_section: CrossSectionSpec = "strip",
    straight_to_grating_spacing: float = 10.0,
    centered: bool = True,
    radius: float | None = None,
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
        straight_to_grating_spacing: spacing between the last grating coupler and the loopback.
        centered: if True, centers the array around the origin.
        radius: optional radius for routing the loopback.
    """
    c = Component()
    grating_coupler = gf.get_component(grating_coupler)
    ports: dict[str, kf.DPort] = {}

    for i in range(n):
        gc = c << grating_coupler
        gc.rotate(rotation)
        gc.x = (i - (n - 1) / 2) * pitch if centered else i * pitch
        port_name_new = f"o{i}"
        ports[port_name_new] = gc.ports[port_name]
        if not with_loopback or i not in [0, n - 1]:
            c.add_port(port=gc.ports[port_name], name=port_name_new)

    if with_loopback:
        if rotation != -90:
            raise ValueError(
                f"with_loopback works only with rotation = -90, got {rotation=}"
            )
        routing_xs = gf.get_cross_section(cross_section)
        radius = radius or routing_xs.radius

        port0 = ports["o0"]
        port1 = ports[f"o{n - 1}"]
        assert radius is not None
        radius_dbu = round(radius / c.kcl.dbu)
        d_loop_um = straight_to_grating_spacing + max(
            [
                grating_coupler.ysize,
                grating_coupler.xsize,
            ]
        )
        d_loop = round(d_loop_um / c.kcl.dbu) + radius_dbu
        waypoints = kf.routing.optical.route_loopback(
            port0.to_itype(),
            port1.to_itype(),
            bend90_radius=radius_dbu,
            d_loop=d_loop,
        )

        waypoints_ = [point.to_dtype(c.kcl.dbu) for point in waypoints]

        gf.routing.route_single(
            c,
            port1=port0,
            port2=port1,
            waypoints=waypoints_,
            cross_section=cross_section,
        )

    return c


if __name__ == "__main__":
    c = grating_coupler_array(
        with_loopback=False, centered=True, cross_section="rib_bbox", n=2
    )
    c.pprint_ports()
    c.show()
