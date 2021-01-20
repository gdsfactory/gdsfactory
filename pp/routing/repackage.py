from typing import Optional

import numpy as np

import pp
from pp.component import Component
from pp.components.bezier import bezier
from pp.container import container


@container
def package_optical2x2(
    component: Component,
    port_spacing: float = 20.0,
    bend_length: Optional[float] = None,
):
    """returns component with port_spacing"""

    comp = component() if callable(component) else component

    if bend_length is None:
        bend_length = port_spacing
    dx = bend_length
    dy = port_spacing / 2.0

    p_w0 = comp.ports["W0"].midpoint
    p_e0 = comp.ports["E0"].midpoint
    p_w1 = comp.ports["W1"].midpoint
    p_e1 = comp.ports["E1"].midpoint

    c = pp.Component(f"{comp.name}_{int(port_spacing)}")
    c << comp

    t = np.linspace(0, 1, 101)

    y0 = p_e1[1]
    control_points = [(0, 0), (dx / 2, 0), (dx / 2, dy - y0), (dx, dy - y0)]

    bezier_bend_t = bezier(
        control_points=control_points, t=t, start_angle=0, end_angle=0
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

    c.info["min_bend_radius"] = bezier_bend_t.info["min_bend_radius"]

    for pname, p in c.ports.items():
        if p.port_type != "optical":
            c.add_port(pname, port=p)

    return c


if __name__ == "__main__":
    from pp.components import coupler

    # component = coupler(gap=0.3)
    # component = pp.c.mzi2x2(with_elec_connections=True)

    c = package_optical2x2(coupler)
    print(c.ports["E1"].y - c.ports["E0"].y)
    print(c.ports)
    pp.show(c)
