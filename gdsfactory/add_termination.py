from functools import partial
from typing import List, Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.port import Port
from gdsfactory.types import ComponentSpec

terminator_function = partial(taper_function, width2=0.1)


@cell
def add_termination(
    component: Component,
    ports: Optional[List[Port]] = None,
    terminator: ComponentSpec = terminator_function,
    port_name: Optional[str] = None,
    port_type: str = "optical",
    **kwargs
) -> Component:
    """Returns component with terminator on some ports.

    Args:
        component: to add terminator.
        ports: optional list of ports to terminate (defaults to all).
        terminator: factory for the terminator.
        port_name: for the terminator to connect to the component ports.
        port_type: of the ports that you want to terminate.
        kwargs: for the ports you want to terminate (orientation, width).
    """
    terminator = gf.get_component(terminator)
    port_name = port_name or terminator.get_ports_list()[0].name

    c = Component()
    c.add_ref(component)
    c.component = component

    ports_all = component.get_ports_list()
    ports = ports or component.get_ports_list(port_type=port_type, **kwargs)

    for port in ports_all:
        if port in ports:
            t_ref = c.add_ref(terminator)
            t_ref.connect(port_name, port)
        else:
            c.add_port(port.name, port=port)

    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = gf.components.straight(length=50)
    cc = add_termination(component=c)
    # cc = add_termination(component=c, orientation=0)
    cc.show(show_ports=True)
