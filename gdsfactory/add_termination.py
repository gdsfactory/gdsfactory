from functools import partial
from typing import List, Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.port import Port
from gdsfactory.types import ComponentFactory

terminator_function = partial(taper_function, width2=0.1)


@cell
def add_termination(
    component: Component,
    ports: Optional[List[Port]] = None,
    terminator: ComponentFactory = terminator_function,
    port_name: Optional[str] = None,
    port_type: str = "optical",
    **kwargs
) -> Component:
    """Returns component containing a comonent with all ports terminated

    Args:
        component:
        terminator: factory for the terminator
        port_name: for the terminator to connect to the component ports
        port_type:
        **kwargs
    """
    terminator = terminator() if callable(terminator) else terminator
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

    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight_heater_metal(length=50)
    cc = add_termination(component=c, orientation=0)
    cc.show()
