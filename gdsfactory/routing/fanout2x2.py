from typing import Callable, Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.port import select_ports_optical


@cell
def fanout2x2(
    component: Component,
    port_spacing: float = 20.0,
    bend_length: Optional[float] = None,
    npoints: int = 101,
    select_ports: Callable = select_ports_optical,
) -> Component:
    """returns component with port_spacing.

    Args:
        component: to package
        port_spacing: for the returned component
        bend_length: length of the bend (defaults to port_spacing)
        npoints: for sbend
        select_ports: function to select  optical_ports ports

    """

    c = gf.Component()

    component = component() if callable(component) else component
    comp = c << component
    comp.movey(-comp.y)

    if bend_length is None:
        bend_length = port_spacing
    dx = bend_length

    y = port_spacing / 2.0

    p_w0 = comp.ports[1].midpoint
    p_w1 = comp.ports[2].midpoint
    p_e1 = comp.ports[3].midpoint
    p_e0 = comp.ports[4].midpoint
    y0 = p_e1[1]

    dy = y - y0

    control_points = [(0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)]

    bezier_bend_t = bezier(
        control_points=control_points, npoints=npoints, start_angle=0, end_angle=0
    )

    b_tr = bezier_bend_t.ref(port_id=1, position=p_e1)
    b_br = bezier_bend_t.ref(port_id=1, position=p_e0, v_mirror=True)
    b_tl = bezier_bend_t.ref(port_id=1, position=p_w1, h_mirror=True)
    b_bl = bezier_bend_t.ref(port_id=1, position=p_w0, rotation=180)

    c.add([b_tr, b_br, b_tl, b_bl])

    c.add_port(1, port=b_bl.ports[2])
    c.add_port(2, port=b_tl.ports[2])
    c.add_port(3, port=b_tr.ports[2])
    c.add_port(4, port=b_br.ports[2])

    c.min_bend_radius = bezier_bend_t.info["min_bend_radius"]

    optical_ports = select_ports(comp.ports)
    for port_name in comp.ports.keys():
        if port_name not in optical_ports:
            c.add_port(f"DC_{port_name}", port=comp.ports[port_name])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # c =gf.components.coupler(gap=1.0)
    c = gf.components.nxn(west=2, east=2)

    cc = fanout2x2(component=c)
    print(cc.ports[3].y - cc.ports[4].y)
    # print(cc.ports)
    cc.show()
