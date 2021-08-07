from typing import Callable

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.electrical.pad import pad_array
from gdsfactory.routing.get_route import get_route_from_waypoints
from gdsfactory.routing.get_route_electrical_shortest_path import (
    get_route_electrical_shortest_path,
)


@gf.cell
def add_electrical_pads_top(
    component: Component,
    dy: float = 100.0,
    route_filter: Callable = get_route_from_waypoints,
    **kwargs,
) -> Component:
    """Returns new component with electrical ports connected to top pad array

    Args:
        component:
        dy: vertical spacing
        kwargs:pad settings
            pad: pad element
            pitch: x spacing
            n: number of pads
            port_list: list of port orientations (N, S, W, E) per pad
            pad_settings: settings for pad if pad is callable
            **port_settings
    """
    c = Component()
    ports = component.get_ports_list(port_type="dc")
    c << component
    pads = c << pad_array(n=len(ports), port_list=["S"], **kwargs)
    pads.x = component.x
    pads.ymin = component.ymax + dy
    ports_pads = list(pads.ports.values())

    ports_pads = gf.routing.sort_ports.sort_ports_x(ports_pads)
    ports_component = gf.routing.sort_ports.sort_ports_x(ports)

    for p1, p2 in zip(ports_component, ports_pads):
        c.add(get_route_electrical_shortest_path(p1, p2))

    c.ports = component.ports.copy()
    for port in ports:
        c.ports.pop(port.name)
    return c


def demo_mzi():
    import gdsfactory as gf

    c = gf.components.straight_with_heater()
    c = gf.components.mzi_phase_shifter()
    cc = add_electrical_pads_top(component=c)
    return cc


def demo_straight():
    import gdsfactory as gf

    c = gf.components.straight_with_heater(
        port_orientation_input=0, port_orientation_output=180
    )
    cc = add_electrical_pads_top(component=c)
    return cc


if __name__ == "__main__":
    # c = demo_mzi()
    # c = demo_straight()
    # c.show()
    import gdsfactory as gf

    c = gf.components.straight_with_heater(
        port_orientation_input=0, port_orientation_output=180
    )
    cc = add_electrical_pads_top(component=c)
    cc.show()
