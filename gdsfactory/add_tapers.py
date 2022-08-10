from typing import Callable, List, Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.port import Port, select_ports_optical
from gdsfactory.types import ComponentSpec


def get_ports_and_tapers(
    component: ComponentSpec,
    taper: ComponentSpec = taper_function,
    select_ports: Optional[Callable] = select_ports_optical,
) -> Tuple[List[Port], List[ComponentReference]]:
    """Returns ports and taper elements for a component."""
    elements = []

    taper = gf.call_if_func(taper)
    component = gf.pdk.get_component(component)
    ports = select_ports(component.ports) if select_ports else component.ports

    for port in component.ports.copy().values():
        if port.name in ports.key():
            taper_ref = taper.ref()
            taper_ref.connect(taper_ref.ports["o2"].name, port)
            elements.append(taper_ref)
            ports.append(taper_ref.ports["o1"])
    return ports, elements


@cell
def add_tapers(
    component: ComponentSpec,
    taper: ComponentSpec = taper_function,
    select_ports: Optional[Callable] = select_ports_optical,
    taper_port_name1: str = "o1",
    taper_port_name2: str = "o2",
) -> Component:
    """Returns new component with taper in all optical ports.

    Args:
        component: spec for the component to add tapers to.
        taper: taper spec for each port.
        select_ports: function to select ports.
        taper_port_name1: for input.
        taper_port_name2: for output.
    """
    c = gf.Component()
    component = gf.pdk.get_component(component)

    ports_to_taper = select_ports(component.ports) if select_ports else component.ports
    ports_to_taper_names = [p.name for p in ports_to_taper.values()]

    for port_name, port in component.ports.items():
        if port.name in ports_to_taper_names:
            taper_ref = c << taper(width2=port.width)
            taper_ref.connect(taper_ref.ports[taper_port_name2].name, port)
            c.add_port(name=port_name, port=taper_ref.ports[taper_port_name1])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # t = gf.components.taper(width2=2)
    # c0 = gf.components.straight_heater_metal(width=2)
    c0 = gf.components.straight(width=2)
    c1 = add_tapers(c0)
    c1.show()

    # c2 = gf.routing.add_fiber_single(c1, with_loopback=False)
    # c2.show()

    # print(cc.ports.keys())
    # print(cc.settings.keys())
    # cc.show(show_ports=True)

    # ports, elements = add_taper_elements(component=c, taper=t)
    # c.ports = ports
    # c.add(elements)
    # c.show(show_ports=True)
    # print(c.ports)
