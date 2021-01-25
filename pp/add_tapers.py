from typing import Callable

import pp
from pp.component import Component
from pp.components.taper import taper as taper_function
from pp.container import container


def add_taper_elements(
    component: Component, taper: Callable = taper_function
) -> Component:
    """returns ports and taper elements for a component"""
    ports = []
    elements = []

    taper_object = pp.call_if_func(taper)
    for port_name, port in component.ports.copy().items():
        if port.port_type == "optical":
            taper_ref = taper_object.ref()
            taper_ref.connect(taper_ref.ports["2"].name, port)
            elements.append(taper_ref)
            ports.append(taper_ref.ports["1"])
    return ports, elements


@container
@pp.port.deco_rename_ports
def add_tapers(
    component: Component,
    taper: Callable = taper_function,
    suffix: str = "t",
    port_type: str = "optical",
) -> Component:
    """returns component optical tapers for component """

    taper_object = pp.call_if_func(taper)
    c = pp.Component(name=f"{component.name}_{suffix}")

    for port_name, port in component.ports.copy().items():
        if port.port_type == port_type:
            taper_ref = c << taper_object
            taper_ref.connect(taper_ref.ports["2"].name, port)
            c.add_port(name=port_name, port=taper_ref.ports["1"])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    return c


if __name__ == "__main__":
    component = pp.c.waveguide(width=2)
    # t = pp.c.taper(width2=2)

    # cc = add_tapers(component=component, taper=t, suffix="t")
    # print(cc.ports.keys())
    # print(cc.settings.keys())
    # pp.show(cc)

    # ports, elements = add_taper_elements(component=c, taper=t)
    # c.ports = ports
    # c.add(elements)
    # pp.show(c)
    # print(c.ports)
