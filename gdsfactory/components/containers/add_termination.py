from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.tapers.taper import taper
from gdsfactory.typings import ComponentSpec

_terminator_function = partial(taper, width2=0.1)


@gf.cell_with_module_name
def add_termination(
    component: ComponentSpec = "straight",
    port_names: tuple[str, ...] | None = None,
    terminator: ComponentSpec = _terminator_function,
    terminator_port_name: str | None = None,
) -> Component:
    """Returns component with terminator on some ports.

    Args:
        component: to add terminator.
        port_names: ports to add terminator.
        terminator: factory for the terminator.
        terminator_port_name: for the terminator to connect to the component ports.
    """
    terminator = gf.get_component(terminator)
    terminator_port_name = terminator_port_name or terminator.ports[0].name

    assert terminator_port_name is not None

    c = Component()
    component = gf.get_component(component)
    ref = c.add_ref(component)

    ports_names_all = [p.name for p in component.ports]
    ports_names_to_terminate = port_names or ports_names_all

    for port_name in ports_names_all:
        if port_name in ports_names_to_terminate:
            t_ref = c.add_ref(terminator)
            t_ref.connect(port=terminator_port_name, other=ref[port_name])
        else:
            port = ref[port_name]
            c.add_port(name=port.name, port=port)

    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = gf.components.straight(length=50)
    cc = add_termination(component=c, terminator_port_name="o1", port_names=("o2",))
    # cc = add_termination(component=c, orientation=0)
    cc.pprint_ports()
    cc.show()
