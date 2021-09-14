from typing import Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.types import ComponentFactory


@cell
def add_termination(
    component: Component,
    terminator: ComponentFactory = taper_function,
    port_name: Optional[str] = None,
) -> Component:
    """Returns component containing a comonent with all ports terminated

    Args:
        component:
        terminator: factory for the terminator
        port_name: for the terminator to connect to the component ports
    """
    terminator = terminator() if callable(terminator) else terminator
    port_name = port_name or terminator.get_ports_list()[0].name

    c = Component()
    c.add_ref(component)
    c.component = component

    for port in component.ports.values():
        t_ref = c.add_ref(terminator)
        t_ref.connect(port_name, port)

    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    terminator = gf.partial(gf.c.taper, width2=0.1)
    cc = add_termination(component=c, terminator=terminator)
    cc.show()
