from __future__ import annotations

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
def loss_deembedding_ch13_24(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    port_name: str = "o1",
    rotation: float = -90,
    yspacing: float | None = None,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->3, 2->4

    Args:
        pitch: um.
        grating_coupler: spec.
        cross_section: spec.
        port_name: for the grating_coupler port.
        rotation: degrees.
        yspacing: um.
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)
    c = gf.Component()
    dx = pitch
    gc_ports = []

    for i in range(4):
        g = c << gc
        g.drotate(rotation)
        g.dmovex(i * dx)
        gc_ports.append(g.ports[port_name])

    route_single(
        c,
        gc_ports[0],
        gc_ports[2],
        start_straight_length=40.0,
        cross_section=cross_section,
        **kwargs,
    )

    x = gf.get_cross_section(cross_section, **kwargs)
    radius = x.radius

    p1 = gc_ports[1]
    p3 = gc_ports[3]
    yspacing = yspacing or gc.dysize + 2 * radius
    bend90 = gf.components.bend_euler(cross_section=cross_section, **kwargs)
    points = gf.kf.routing.optical.route_loopback(
        p1,
        p3,
        bend90_radius=round(radius / c.kcl.dbu),
        d_loop=round(yspacing / c.kcl.dbu),
    )
    route_single(
        c,
        port1=p1,
        port2=p3,
        waypoints=points,
        bend=bend90,
        straight=gf.components.straight,
        cross_section=cross_section,
        **kwargs,
    )

    return c


@cell
def loss_deembedding_ch12_34(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = "grating_coupler_te",
    port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    rotation: float = -90,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->2, 3->4

    Args:
        pitch: um.
        grating_coupler: spec.
        port_name: for the grating_coupler port.
        cross_section: spec.
        rotation: degrees.
        kwargs: cross_section settings.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch

    gc_ports = []

    for i in range(4):
        g = c << gc
        g.drotate(rotation)
        g.dmove((i * dx, 0))
        gc_ports.append(g.ports[port_name])

    route_single(
        c,
        gc_ports[0],
        gc_ports[1],
        start_straight_length=40.0,
        cross_section=cross_section,
        **kwargs,
    )
    route_single(
        c,
        gc_ports[2],
        gc_ports[3],
        start_straight_length=40.0,
        cross_section=cross_section,
        **kwargs,
    )
    return c


@cell
def loss_deembedding_ch14_23(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = "grating_coupler_te",
    cross_section: CrossSectionSpec = "strip",
    port_name: str = "o1",
    rotation: float = -90,
    **kwargs,
) -> Component:
    """Grating coupler test structure for fiber array.

    Connects channel 1->4, 2->3

    Args:
        pitch: um.
        grating_coupler: spec.
        cross_section: spec.
        port_name: for the grating_coupler port.
        rotation: degrees.
        kwargs: cross_section settings.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch
    gc_ports = []

    for i in range(4):
        g = c << gc
        g.drotate(rotation)
        g.dmove((i * dx, 0))
        gc_ports.append(g.ports[port_name])

    route_single(
        c,
        gc_ports[0],
        gc_ports[3],
        start_straight_length=40.0,
        cross_section=cross_section,
        **kwargs,
    )
    route_single(
        c,
        gc_ports[1],
        gc_ports[2],
        start_straight_length=30.0,
        cross_section=cross_section,
        **kwargs,
    )
    return c


@cell
def grating_coupler_loss_fiber_array(
    pitch: float = 127.0,
    grating_coupler: ComponentSpec = "grating_coupler_te",
    port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    rotation: float = -90,
    **kwargs,
) -> Component:
    """Returns Grating coupler fiber array loopback.

    Args:
        pitch: spacing.
        grating_coupler: spec for grating coupler.
        port_name: for the grating_coupler port.
        cross_section: spec.
        rotation: degrees.
        kwargs: cross_section settings.

    Keyword Args:
        kwargs: cross_section settings.
    """
    gc = gf.get_component(grating_coupler)

    c = gf.Component()
    dx = pitch
    gc_ports = []

    for i in range(2):
        g = c << gc
        g.drotate(rotation)
        g.dmove((i * dx, 0))
        gc_ports.append(g.ports[port_name])

    route_single(
        c,
        gc_ports[0],
        gc_ports[1],
        start_straight_length=40.0,
        cross_section=cross_section,
        **kwargs,
    )
    return c


@cell
def grating_coupler_loss_fiber_array4(
    pitch: float = 127.0, grating_coupler: ComponentSpec = grating_coupler_te, **kwargs
) -> Component:
    """Returns a grating coupler test structure for fiber array.

    Measures all combinations for a 4 fiber fiber_array

    Connects channel 1->3, 2->4
    Connects channel 1->4, 2->3
    Connects channel 1->2, 3->4

    Args:
        pitch: grating_coupler_pitch.
        grating_coupler: function.
        kwargs: cross_section settings.
    """
    c = gf.Component()
    c1 = loss_deembedding_ch13_24(grating_coupler=grating_coupler, **kwargs)
    c2 = loss_deembedding_ch14_23(grating_coupler=grating_coupler, **kwargs)
    c3 = loss_deembedding_ch12_34(grating_coupler=grating_coupler, **kwargs)
    c.add_ref(c1)
    c2 = c.add_ref(c2)
    c3 = c.add_ref(c3)
    c2.dmovex(pitch * 4)
    c3.dmovex(pitch * 8)
    return c


if __name__ == "__main__":
    # c = loss_deembedding_ch14_23()
    # c = loss_deembedding_ch12_34()
    c = loss_deembedding_ch13_24()
    # c = grating_coupler_loss_fiber_array4()
    # c = grating_coupler_loss_fiber_array4(layer=(2, 0), radius=30)
    # c = grating_coupler_loss_fiber_array4(cross_section="rib")
    # c = grating_coupler_loss_fiber_array(layer=(2, 0), radius=30)
    # c = grating_coupler_loss_fiber_array()
    c.show()
