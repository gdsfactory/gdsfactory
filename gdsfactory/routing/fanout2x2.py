from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier


@cell
def fanout2x2(
    component: Component,
    port_spacing: float = 20.0,
    bend_length: Optional[float] = None,
    npoints: int = 101,
) -> Component:
    """returns component with port_spacing.

    Args:
        component: to package
        port_spacing: for the returned component
        bend_length: length of the bend (defaults to port_spacing)
        npoints: for sbend

    """

    c = gf.Component()

    component = component() if callable(component) else component
    comp = c << component
    comp.movey(-comp.y)

    if bend_length is None:
        bend_length = port_spacing
    dx = bend_length

    y = port_spacing / 2.0

    p_w0 = comp.ports["W0"].midpoint
    p_e0 = comp.ports["E0"].midpoint
    p_w1 = comp.ports["W1"].midpoint
    p_e1 = comp.ports["E1"].midpoint
    y0 = p_e1[1]

    dy = y - y0

    control_points = [(0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)]

    bezier_bend_t = bezier(
        control_points=control_points, npoints=npoints, start_angle=0, end_angle=0
    )

    b_tr = bezier_bend_t.ref(port_id="0", position=p_e1)
    b_br = bezier_bend_t.ref(port_id="0", position=p_e0, v_mirror=True)
    b_tl = bezier_bend_t.ref(port_id="0", position=p_w1, h_mirror=True)
    b_bl = bezier_bend_t.ref(port_id="0", position=p_w0, rotation=180)

    c.add([b_tr, b_br, b_tl, b_bl])

    c.add_port("W0", port=b_bl.ports["1"])
    c.add_port("W1", port=b_tl.ports["1"])
    c.add_port("E0", port=b_br.ports["1"])
    c.add_port("E1", port=b_tr.ports["1"])

    c.min_bend_radius = bezier_bend_t.info["min_bend_radius"]

    for pname, p in c.ports.items():
        if p.port_type != "optical":
            c.add_port(pname, port=p)

    return c


if __name__ == "__main__":
    # c =gf.components.coupler(gap=1.0)
    c = gf.components.nxn(west=4)

    cc = fanout2x2(component=c)
    print(cc.ports["E1"].y - cc.ports["E0"].y)
    # print(cc.ports)
    cc.show()
