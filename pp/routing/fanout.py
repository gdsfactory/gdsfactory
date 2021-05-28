from typing import List, Tuple

import pp
from pp.cell import cell
from pp.component import Component
from pp.port import auto_rename_ports, port_array
from pp.routing.routing import route_basic


@cell
def fanout_componnet(
    component: Component,
    ports: List[pp.Port],
    pitch: Tuple[float, float] = (0.0, 20.0),
    dx: float = 20.0,
    rename_ports: bool = True,
    **kwargs,
) -> Component:
    """Returns component with Sbend fanout routes.

    Args:
        component: to package
        pitch: target port spacing for new component
        dx: how far the fanout
        rename_ports
        kwargs: for route_basic
    """

    c = Component()
    comp = component() if callable(component) else component
    comp.movey(-comp.y)
    c.add_ref(comp)

    c.ports = comp.ports.copy()
    ports1 = ports
    port = ports1[0]
    port_extended_x = port.get_extended_midpoint(dx)[0]
    port_settings = port.settings.copy()

    port_settings.pop("name")
    port_settings.update(midpoint=(port_extended_x, 0))
    port_settings.update(orientation=(port.angle + 180) % 360)
    ports2 = port_array(n=len(ports1), pitch=pitch, **port_settings)

    for i, (p1, p2) in enumerate(zip(ports1, ports2)):
        path = route_basic(port1=p1, port2=p2, **kwargs)
        path_ref = path.ref()
        c.add(path_ref)
        c.ports.pop(p1.name)
        c.add_port(f"{i}", port=p2)

    if rename_ports:
        auto_rename_ports(c)
    return c


def fanout_ports(
    ports: List[pp.Port],
    pitch: Tuple[float, float] = (0.0, 20.0),
    dx: float = 20.0,
    **kwargs,
) -> List[pp.types.Route]:
    """Returns fanout Sbend routes.

    Args:
        ports: list of ports
        pitch: target port spacing for new component
        dx: how far the fanout
    """
    routes = []
    ports1 = ports
    port = ports1[0]
    port_extended_x = port.get_extended_midpoint(dx)[0]
    port_settings = port.settings.copy()

    port_settings.pop("name")
    port_settings.update(midpoint=(port_extended_x, 0))
    port_settings.update(orientation=(port.angle + 180) % 360)
    ports2 = port_array(n=len(ports1), pitch=pitch, **port_settings)

    for p1, p2 in zip(ports1, ports2):
        route = pp.routing.get_route_sbend(p1, p2)
        routes.append(route)
    return routes


if __name__ == "__main__":
    # c = pp.components.mzi2x2(with_elec_connections=True)
    # c =pp.components.coupler(gap=1.0)
    # c = pp.c.nxn(west=4)
    c = pp.c.nxn(west=4, layer=pp.LAYER.SLAB90)

    cc = fanout_componnet(component=c, ports=c.get_ports_list(orientation=180))
    print(len(cc.ports))
    cc.show(show_ports=True)

    # c = pp.c.nxn(west=4, layer=pp.LAYER.SLAB90)
    # routes = fanout_ports(ports=c.get_ports_list(orientation=180))

    # for route in routes:
    #     c.add(route.references)
    # c.show(show_ports=True)
