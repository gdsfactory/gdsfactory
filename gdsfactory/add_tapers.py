from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.port import Port, select_ports_optical
from gdsfactory.typings import ComponentFactory, ComponentSpec


def get_ports_and_tapers(
    component: ComponentSpec,
    taper: ComponentSpec = taper_function,
    select_ports: Callable | None = select_ports_optical,
) -> tuple[list[Port], list[ComponentReference]]:
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
    taper: ComponentFactory = taper_function,
    select_ports: Callable | None = select_ports_optical,
    ports: list[Port] | None = None,
    taper_port_name1: str = "o1",
    taper_port_name2: str = "o2",
) -> Component:
    """Returns new component with taper in all optical ports.

    Args:
        component: spec for the component to add tap ers to.
        taper: taper spec for each port.
        select_ports: function to select ports.
        ports: Optional list of ports to add tapers to. Defaults to select_ports(component.ports).
        taper_port_name1: for input.
        taper_port_name2: for output.
    """
    c = gf.Component()
    component = gf.get_component(component)

    if select_ports and not callable(select_ports):
        raise ValueError(f"select_ports should be a function, got {type(select_ports)}")

    if not callable(taper):
        raise ValueError(f"taper should be a function, got {type(taper)}")

    ports_to_taper = (
        ports or select_ports(component.ports).values()
        if select_ports
        else component.ports
    )
    ports_to_taper_names = [p.name for p in ports_to_taper]

    for port_name, port in component.ports.items():
        if port.name in ports_to_taper_names:
            _taper = taper(width2=port.width)
            taper_ref = c << _taper
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
