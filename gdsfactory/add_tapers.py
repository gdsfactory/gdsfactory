from typing import List, Tuple

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.port import Port, auto_rename_ports
from gdsfactory.types import ComponentFactory


def add_taper_elements(
    component: Component,
    taper: ComponentFactory = taper_function,
    taper_port_name1: str = "1",
    taper_port_name2: str = "2",
    port_type: str = "optical",
) -> Tuple[List[Port], List[ComponentReference]]:
    """Returns ports and taper references."""
    ports = []
    elements = []

    taper_component = taper()
    if taper_port_name1 not in taper_component.ports:
        raise ValueError(
            f"{taper_port_name1} not in {list(taper_component.ports.keys())}"
        )
    if taper_port_name2 not in taper_component.ports:
        raise ValueError(
            f"{taper_port_name2} not in {list(taper_component.ports.keys())}"
        )

    for port in component.ports.copy().values():
        if port.port_type == port_type:
            taper_ref = taper_component.ref()
            taper_ref.connect(taper_ref.ports[taper_port_name2].name, port)
            elements.append(taper_ref)
            ports.append(taper_ref.ports[taper_port_name1])
    return ports, elements


@gf.cell
def add_tapers(
    component: Component,
    taper: ComponentFactory = taper_function,
    port_type: str = "optical",
    waveguide="strip",
    taper_port_name1: str = "1",
    taper_port_name2: str = "2",
    with_auto_rename: bool = False,
) -> Component:
    """Returns component with tapers."""

    taper_component = taper()
    if taper_port_name1 not in taper_component.ports:
        raise ValueError(
            f"{taper_port_name1} not in {list(taper_component.ports.keys())}"
        )
    if taper_port_name2 not in taper_component.ports:
        raise ValueError(
            f"{taper_port_name2} not in {list(taper_component.ports.keys())}"
        )
    c = gf.Component()

    for port_name, port in component.ports.items():
        if port.port_type == port_type:
            taper_ref = c << taper_component
            taper_ref.connect(taper_ref.ports[taper_port_name2].name, port)
            c.add_port(name=port_name, port=taper_ref.ports[taper_port_name1])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    if with_auto_rename:
        auto_rename_ports(c)
    return c


if __name__ == "__main__":
    from functools import partial

    c0 = gf.components.straight(width=2)

    t = partial(taper_function, width2=2)
    c1 = add_tapers(component=c0, taper=t)
    c1.show()

    # print(cc.ports.keys())
    # print(cc.settings.keys())
    # cc.show()

    # ports, elements = add_taper_elements(component=c, taper=t)
    # c.ports = ports
    # c.add(elements)
    # c.show()
    # print(c.ports)
