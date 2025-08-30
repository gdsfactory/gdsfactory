from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Strs


@gf.cell(set_name=False)
def extend_ports_list(
    component_spec: ComponentSpec,
    extension: ComponentSpec,
    extension_port_name: str | None = None,
    ignore_ports: Strs | None = None,
) -> Component:
    """Returns a component with an extension attached to a list of ports.

    Args:
        component_spec: component from which to get ports.
        extension: function for extension.
        extension_port_name: to connect extension.
        ignore_ports: list of port names to ignore.
    """
    from gdsfactory.pdk import get_component

    ports = get_component(component_spec).ports

    c = Component()
    extension = get_component(extension)
    c.name = f"{extension.name}_extended_{c.cell_index()}"

    extension_port_name_or_port = extension_port_name or extension.ports[0]
    ignore_ports = ignore_ports or ()

    for i, port in enumerate(ports):
        extension_ref = c << extension
        extension_ref.connect(extension_port_name_or_port, port)

        for port in extension_ref.ports:
            port_name = port.name
            if port_name not in ignore_ports:
                c.add_port(f"{i}_{port_name}", port=port)

    c.auto_rename_ports()
    return c
