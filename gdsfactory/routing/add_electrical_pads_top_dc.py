from typing import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array as pad_array_function
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.get_bundle import get_bundle_electrical
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.types import ComponentSpec, Float2


@cell
def add_electrical_pads_top_dc(
    component: ComponentSpec = straight_heater_metal,
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentSpec = pad_array_function,
    select_ports: Callable = select_ports_electrical,
    get_bundle_function: Callable = get_bundle_electrical,
    **kwargs,
) -> Component:
    """Returns new component with electrical ports connected to a pad array at \
    the top.

    Args:
        component: component spec to connect to.
        spacing: component to pad spacing.
        pad_array: component spec for pad_array.
        select_ports: function to select_ports.
        get_bundle_function: function to route bundle of ports.
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
    ports = select_ports(cref.ports)
    ports_component = list(ports.values())
    ports_component = [port.copy() for port in ports_component]

    for port in ports_component:
        port.orientation = 90

    pad_array = gf.get_component(pad_array, columns=len(ports))
    pads = c << pad_array
    pads.x = cref.x + spacing[0]
    pads.ymin = cref.ymax + spacing[1]

    ports_pads = list(pads.ports.values())
    ports_component = sort_ports_x(ports_component)
    ports_pads = sort_ports_x(ports_pads)

    routes = get_bundle_function(ports_component, ports_pads, **kwargs)
    for route in routes:
        c.add(route.references)

    c.add_ports(cref.ports)

    # remove electrical ports
    for port in ports_component:
        c.ports.pop(port.name)

    c.add_ports(pads.ports)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = gf.components.straight_heater_metal(length=100.0)
    c = gf.components.straight(length=100.0)
    cc = add_electrical_pads_top_dc(component=c, width=10)
    cc.show(show_ports=True)
