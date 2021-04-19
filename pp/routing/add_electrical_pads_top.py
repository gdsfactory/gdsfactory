from typing import Callable

from pp.cell import cell
from pp.component import Component
from pp.components.electrical.pad import pad_array
from pp.routing.get_route import get_route_from_waypoints_electrical
from pp.routing.get_route_electrical_shortest_path import (
    get_route_electrical_shortest_path,
)


@cell
def add_electrical_pads_top(
    component: Component,
    component_top_to_pad_bottom_distance: float = 100.0,
    route_filter: Callable = get_route_from_waypoints_electrical,
    **kwargs,
) -> Component:
    """connects component electrical ports with pad array at the top

    Args:
        component:
        pad: pad element
        spacing: pad array (x, y) spacing
        width: pad width
        height: pad height
        layer: pad layer
    """
    c = Component(f"{component.name}_e")
    ports = component.get_ports_list(port_type="dc")
    # for port in ports:
    #     print(port.name)
    # print(len(ports))
    c << component
    pads = c << pad_array(n=len(ports), port_list=["S"], **kwargs)
    pads.x = component.x
    pads.ymin = component.ymax + component_top_to_pad_bottom_distance

    ports_pads = list(pads.ports.values())

    ports_pads.sort(key=lambda p: p.x)
    ports.sort(key=lambda p: p.x)

    for p1, p2 in zip(ports_pads, ports):
        c.add(get_route_electrical_shortest_path(p1, p2))

    c.ports = component.ports.copy()
    for port in ports:
        c.ports.pop(port.name)
    return c


if __name__ == "__main__":
    import pp

    c = pp.components.mzi2x2(with_elec_connections=True)
    c = pp.components.straight_with_heater()
    cc = add_electrical_pads_top(component=c)
    cc.show()
