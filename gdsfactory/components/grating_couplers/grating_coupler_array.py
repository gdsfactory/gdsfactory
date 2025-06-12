from __future__ import annotations

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.auto_taper import add_auto_tapers
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
    bend: ComponentSpec = "bend_euler",
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
        bend: ComponentSpec for the bend used in the loopback.
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
        if radius is None:
            bend_component = gf.get_component(bend, cross_section=cross_section)
            try:
                radius = _get_routing_radius(bend_component, cross_section)
                bend = bend_component
            except KeyError:
                raise ValueError(
                    "Radius must be set in the cross_section or bend component if not provided explicitly."
                )

        port0 = ports["o0"]
        port1 = ports[f"o{n - 1}"]
        assert radius is not None
        radius_dbu = c.kcl.to_dbu(radius)
        d_loop_um = straight_to_grating_spacing + max(
            [
                grating_coupler.ysize,
                grating_coupler.xsize,
            ]
        )
        d_loop = c.kcl.to_dbu(d_loop_um) + radius_dbu

        port0 = add_auto_tapers(c, [port0], cross_section)[0]
        port1 = add_auto_tapers(c, [port1], cross_section)[0]
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
            radius=radius,
            bend=bend,
        )

    return c


def _get_routing_radius(bend: Component, cross_section: CrossSectionSpec) -> float:
    """Get the routing radius from the bend component for ports on the given cross_section."""
    cs = gf.get_cross_section(cross_section)
    cs_layer = cs.layer
    bend_ports = [p for p in bend.ports if p.layer == cs_layer]

    if len(bend_ports) != 2:
        raise ValueError(
            f"Expected 2 ports in bend component with layer {cs_layer}, got {len(bend_ports)}"
        )

    p1, p2 = bend_ports
    dx = abs(p1.center[0] - p2.center[0])
    dy = abs(p1.center[1] - p2.center[1])
    if dx != dy:
        raise ValueError(
            f"Expected ports to have equal spacing in x and y, got dx={dx} and dy={dy}"
        )
    return dx


if __name__ == "__main__":
    c = grating_coupler_array(
        with_loopback=False, centered=True, cross_section="rib_bbox", n=2
    )
    c.pprint_ports()
    c.show()
