from typing import Callable, Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.port import select_ports_optical


@gf.cell
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
    component.component = component
    ref = c << component
    ref.movey(-ref.y)

    if bend_length is None:
        bend_length = port_spacing
    dx = bend_length

    y = port_spacing / 2.0

    p_w0 = ref.ports["o1"].midpoint
    p_w1 = ref.ports["o2"].midpoint
    p_e1 = ref.ports["o3"].midpoint
    p_e0 = ref.ports["o4"].midpoint
    y0 = p_e1[1]

    dy = y - y0

    control_points = [(0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)]

    bezier_bend_t = bezier(
        control_points=control_points, npoints=npoints, start_angle=0, end_angle=0
    )

    b_tr = bezier_bend_t.ref(port_id="o1", position=p_e1)
    b_br = bezier_bend_t.ref(port_id="o1", position=p_e0, v_mirror=True)
    b_tl = bezier_bend_t.ref(port_id="o1", position=p_w1, h_mirror=True)
    b_bl = bezier_bend_t.ref(port_id="o1", position=p_w0, rotation=180)

    c.add([b_tr, b_br, b_tl, b_bl])

    c.add_port("o1", port=b_bl.ports["o2"])
    c.add_port("o2", port=b_tl.ports["o2"])
    c.add_port("o3", port=b_tr.ports["o2"])
    c.add_port("o4", port=b_br.ports["o2"])

    c.min_bend_radius = bezier_bend_t.info["min_bend_radius"]

    optical_ports = select_ports(ref.ports)
    for port_name in ref.ports.keys():
        if port_name not in optical_ports:
            c.add_port(port_name, port=ref.ports[port_name])
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # c =gf.components.coupler(gap=1.0)
    c = gf.components.nxn(west=2, east=2)

    cc = fanout2x2(component=c)
    print(cc.ports["o3"].y - cc.ports["o4"].y)
    # print(cc.ports)
    cc.show()
