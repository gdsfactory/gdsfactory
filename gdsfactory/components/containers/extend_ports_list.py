from __future__ import annotations

__all__ = ["extend_ports_list"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Strs


@gf.cell(set_name=False, tags=["containers"])
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
    return cf.extend_ports_list(
        component_spec=component_spec,
        extension=extension,
        extension_port_name=extension_port_name,
        ignore_ports=ignore_ports,
    )
