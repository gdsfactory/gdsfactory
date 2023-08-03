from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.get_bundle import get_bundle_electrical
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.typings import ComponentSpec, Float2, Strs


@cell
def add_electrical_pads_top_dc(
    component: ComponentSpec = straight_heater_metal,
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentSpec = "pad_array",
    select_ports: Callable = select_ports_electrical,
    get_bundle_function: Callable = get_bundle_electrical,
    port_names: Strs | None = None,
    **kwargs,
) -> Component:
    """Returns new component with electrical ports connected to top pad array.

    Args:
        component: component spec to connect to.
        spacing: component to pad spacing.
        pad_array: component spec for pad_array.
        select_ports: function to select_ports.
        get_bundle_function: function to route bundle of ports.
        port_names: optional port names. Overrides select_ports.
        kwargs: route settings.

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.components.straight_heater_metal(length=100)
        c = gf.routing.add_electrical_pads_top_dc(c, width=10)
        c.plot()

    """
    c = Component()
    component = gf.get_component(component)

    cref = c << component
    ports = [cref[port_name] for port_name in port_names] if port_names else None
    ports = ports or select_ports(cref.ports)

    if not ports:
        raise ValueError(
            f"select_ports or port_names did not match any ports in {list(component.ports.keys())}"
        )

    ports_component = list(ports.values()) if isinstance(ports, dict) else ports
    ports_component = [port.copy() for port in ports_component]

    for port in ports_component:
        port.orientation = 90

    pad_array = gf.get_component(pad_array, columns=len(ports))
    pads = c << pad_array
    pads.x = cref.x + spacing[0]
    pads.ymin = cref.ymax + spacing[1]

    ports_pads = pads.get_ports_list(orientation=270)
    ports_component = sort_ports_x(ports_component)
    ports_pads = sort_ports_x(ports_pads)

    routes = get_bundle_function(ports_component, ports_pads, **kwargs)
    for route in routes:
        c.add(route.references)

    c.add_ports(cref.ports)

    # remove electrical ports
    for port in ports_component:
        c.ports.pop(port.name)

    for i, port_pad in enumerate(ports_pads):
        c.add_port(port=port_pad, name=f"elec-{component.name}-{i}")
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = gf.components.straight_heater_metal(length=100.0)
    # c = gf.components.straight(length=100.0)
    cc = add_electrical_pads_top_dc(component=c, width=10)
    cc.show(show_ports=True)
