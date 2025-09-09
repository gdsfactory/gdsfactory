from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_optical
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, PortsFactory


def fanout2x2(
    component: ComponentSpec = "mmi2x2",
    port_spacing: float = 20.0,
    bend_length: float | None = None,
    npoints: int = 101,
    select_ports: PortsFactory = select_ports_optical,
    cross_section: CrossSectionSpec = "strip",
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    port4: str = "o4",
    **kwargs: Any,
) -> Component:
    """Returns component with Sbend fanout routes.

    Args:
        component: to fanout.
        port_spacing: for the returned component.
        bend_length: length of the bend (defaults to port_spacing).
        npoints: for sbend.
        select_ports: function to select  optical_ports ports.
        cross_section: cross_section spec.
        port1: bottom west port.
        port2: top west port.
        port3: top east port.
        port4: bottom east port.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.components.nxn(west=2, east=2)

        cc = gf.routing.fanout2x2(component=c, port_spacing=20)
        cc.plot()

    """
    c = gf.Component()

    component = gf.get_component(component)
    ref = c << component
    ref.movey(-ref.y)

    if bend_length is None:
        bend_length = port_spacing
    dx = bend_length

    y = port_spacing / 2.0

    p_w0 = ref.ports[port1]
    p_w1 = ref.ports[port2]
    p_e1 = ref.ports[port3]
    p_e0 = ref.ports[port4]

    y0 = p_e1.center[1]
    dy = y - y0

    x = gf.get_cross_section(cross_section, **kwargs)
    bend = gf.c.bend_s(size=(dx, dy), npoints=npoints, cross_section=x)

    b_tr = c << bend
    b_br = c << bend
    b_tl = c << bend
    b_bl = c << bend

    b_tr.connect(port=port1, other=p_e1)
    b_br.connect(port=port1, other=p_e0, mirror=True)
    b_tl.connect(port=port1, other=p_w1, mirror=True)
    b_bl.connect(port=port1, other=p_w0)

    c.add_port(port1, port=b_bl.ports[port2])
    c.add_port(port2, port=b_tl.ports[port2])
    c.add_port(port3, port=b_tr.ports[port2])
    c.add_port(port4, port=b_br.ports[port2])

    c.info["min_bend_radius"] = bend.info["min_bend_radius"]

    optical_ports = select_ports(ref.ports)
    optical_port_names = [port.name for port in optical_ports]
    for port in ref.ports:
        port_name = port.name
        if port_name not in optical_port_names:
            c.add_port(port_name, port=ref.ports[port_name])
    c.copy_child_info(component)
    return c
