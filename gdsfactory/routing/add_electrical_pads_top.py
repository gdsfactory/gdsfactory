import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array as pad_array_function
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.get_route_electrical_shortest_path import (
    get_route_electrical_shortest_path,
)
from gdsfactory.types import ComponentFactory, Float2


@gf.cell
def add_electrical_pads_top(
    component: Component,
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentFactory = pad_array_function,
    select_ports=select_ports_electrical,
    **kwargs,
) -> Component:
    """Returns new component with electrical ports connected to top pad array

    Args:
        component:
        spacing: component to pad spacing
        select_ports: function to select electrical ports
        kwargs: pad settings
            pad: pad element
            pitch: x spacing
            n: number of pads
            **port_settings
    """
    c = Component()
    c.component = component
    ref = c << component
    ports = select_ports(ref.ports)
    ports = list(ports.values())
    pads = c << pad_array_function(columns=len(ports), orientation=270, **kwargs)
    pads.x = ref.x + spacing[0]
    pads.ymin = ref.ymax + spacing[1]
    ports_pads = list(pads.ports.values())

    ports_pads = gf.routing.sort_ports.sort_ports_x(ports_pads)
    ports_component = gf.routing.sort_ports.sort_ports_x(ports)

    for p1, p2 in zip(ports_component, ports_pads):
        c.add(get_route_electrical_shortest_path(p1, p2))

    c.add_ports(ref.ports)
    for port in ports:
        c.ports.pop(port.name)
    gf.functions.copy_settings(component, c)
    return c


if __name__ == "__main__":
    # FIXME
    # c = demo_mzi()
    # c = demo_straight()
    # c.show()
    import gdsfactory as gf

    c = gf.components.straight_heater_metal()
    c = gf.components.mzi_phase_shifter()
    cc = add_electrical_pads_top(component=c, spacing=(-150, 30))
    cc.show()
