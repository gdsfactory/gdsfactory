from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_bundle import route_bundle_electrical
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Float2,
    SelectPorts,
    Strs,
)


def add_electrical_pads_top_dc(
    component: ComponentSpec,
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentSpec = "pad_array270",
    select_ports: SelectPorts = select_ports_electrical,
    port_names: Strs | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
    **kwargs: Any,
) -> Component:
    """Returns new component with electrical ports connected to top pad array.

    Args:
        component: component spec to connect to.
        spacing: component to pad spacing.
        pad_array: component factor for pad_array
        select_ports: function to select_ports.
        route_bundle_function: function to route bundle of ports.
        port_names: optional port names. Overrides select_ports.
        cross_section: cross_section for the route.
        kwargs: route settings.

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.components.wire_straight(length=200.)
        c = gf.routing.add_electrical_pads_top_dc(c, width=10)
        c.plot()

    """
    c = Component()
    component = gf.get_component(component)

    cref = c << component
    ports = (
        [cref[port_name] for port_name in port_names]
        if port_names
        else select_ports(cref.ports)
    )

    if not ports:
        raise ValueError(
            f"select_ports or port_names did not match any ports in "
            f"{[port.name for port in component.ports]}"
        )

    ports_component = [port.copy() for port in ports]

    for port in ports_component:
        port.orientation = 90

    pad_array_component = gf.get_component(pad_array, columns=len(ports))
    pads = c << pad_array_component
    pads.x = cref.x + spacing[0]
    pads.ymin = cref.ymax + spacing[1]

    ports_pads = pads.ports.filter(orientation=270)
    ports_component = sort_ports_x(ports_component)
    ports_pads = sort_ports_x(ports_pads)

    route_bundle_electrical(
        c, ports_component, ports_pads, cross_section=cross_section, **kwargs
    )

    for port in cref.ports:
        if port not in ports_component:
            c.add_port(name=port.name, port=port)

    for i, port_pad in enumerate(ports_pads):
        c.add_port(port=port_pad, name=f"elec-{component.name}-{i}")
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = gf.c.wire_straight(length=200.0)
    cc = add_electrical_pads_top_dc(c)
    cc.show()
