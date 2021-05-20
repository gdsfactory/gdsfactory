from typing import Optional, Tuple

import pp
from pp.cell import cell
from pp.component import Component
from pp.port import auto_rename_ports, port_array
from pp.routing.routing import route_basic


@cell
def fanout(
    component: Component,
    port_spacing: float = 20.0,
    delta: Tuple[float, float] = (0.0, 20.0),
    dx: float = 20.0,
    npoints: int = 101,
    prefix: Optional[str] = None,
    port_type: Optional[str] = None,
    layer: Optional[Tuple[int, int]] = None,
    orientation: Optional[float] = None,
    rename_ports: bool = True,
    **kwargs,
) -> Component:
    """Returns component with fanout.

    Args:
        component: to package
        port_spacing: for the returned component
        delta: target port spacing for new component
        dx: how far the fanout
        npoints: for the Sbend
        port_type
        npoints: for sbend
    """

    c = Component()
    comp = component() if callable(component) else component
    comp.movey(-comp.y)
    c.add_ref(comp)

    c.ports = comp.ports.copy()

    ports1 = comp.get_ports_list(
        prefix=prefix, port_type=port_type, layer=layer, orientation=orientation
    )

    port = ports1[0]
    port_extended_x = port.get_extended_midpoint(dx)[0]
    port_settings = port.settings.copy()

    port_settings.pop("name")
    port_settings.update(midpoint=(port_extended_x, 0))
    port_settings.update(orientation=(port.angle + 180) % 360)
    ports2 = port_array(n=len(ports1), delta=delta, **port_settings)

    for i, (p1, p2) in enumerate(zip(ports1, ports2)):
        path = route_basic(port1=p1, port2=p2, **kwargs)
        path_ref = path.ref()
        c.add(path_ref)
        c.ports.pop(p1.name)
        c.add_port(f"{i}", port=p2)

    if rename_ports:
        auto_rename_ports(c)
    return c


if __name__ == "__main__":
    # c = pp.components.mzi2x2(with_elec_connections=True)
    # c =pp.components.coupler(gap=1.0)
    c = pp.c.nxn(west=4)

    cc = fanout(component=c)
    print(len(cc.ports))
    cc.show(show_ports=True)
