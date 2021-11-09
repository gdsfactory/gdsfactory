import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array as pad_array_function
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.types import ComponentFactory, Float2


@gf.cell
def add_electrical_pads_top(
    component: Component,
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentFactory = pad_array_function,
    select_ports=select_ports_electrical,
    layer: gf.types.Layer = (31, 0),
) -> Component:
    """Returns new component with electrical ports connected to top pad array

    Args:
        component:
        spacing: component to pad spacing
        pad_array: function for pad_array
        select_ports: function to select electrical ports
        layer: for the routes
    """
    c = Component()
    c.component = component
    ref = c << component
    ports = select_ports(ref.ports)
    ports = list(ports.values())
    pads = c << pad_array(columns=len(ports), orientation=270)
    pads.x = ref.x + spacing[0]
    pads.ymin = ref.ymax + spacing[1]
    ports_pads = list(pads.ports.values())

    ports_pads = gf.routing.sort_ports.sort_ports_x(ports_pads)
    ports_component = gf.routing.sort_ports.sort_ports_x(ports)

    for p1, p2 in zip(ports_component, ports_pads):
        c.add(route_quad(p1, p2, layer=layer))

    c.add_ports(ref.ports)
    for port in ports:
        c.ports.pop(port.name)
    c.copy_child_info(component)
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
