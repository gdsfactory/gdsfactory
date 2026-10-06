from __future__ import annotations

__all__ = ["add_termination"]


from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.typings import ComponentSpec


def add_termination(
    component: ComponentSpec = "straight",
    port_names: tuple[str, ...] | None = None,
    terminator: ComponentSpec = "taper_terminator",
    terminator_port_name: str | None = None,
) -> Component:
    """Returns component with terminator on some ports.

    Args:
        component: to add terminator.
        port_names: ports to add terminator.
        terminator: factory for the terminator.
        terminator_port_name: for the terminator to connect to the component ports.
    """
    terminator = get_component(terminator)
    terminator_port_name = terminator_port_name or terminator.ports[0].name

    assert terminator_port_name is not None

    c = Component()
    component = get_component(component)
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
