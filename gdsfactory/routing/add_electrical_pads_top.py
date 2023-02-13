from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array as pad_array_function
from gdsfactory.components.straight import straight
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.typings import ComponentSpec, Float2


@gf.cell
def add_electrical_pads_top(
    component: ComponentSpec = straight,
    direction: str = "top",
    spacing: Float2 = (0.0, 100.0),
    pad_array: ComponentSpec = pad_array_function,
    select_ports=select_ports_electrical,
    layer: gf.typings.LayerSpec = "MTOP",
) -> Component:
    """Returns new component with electrical ports connected to top pad array.

    Args:
        component: to route.
        direction: 'top' or 'right', sets direction of the array.
        spacing: component to pad spacing.
        pad_array: function for pad_array.
        select_ports: function to select electrical ports.
        layer: for the routes.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.straight_heater_metal()
        cc = gf.routing.add_electrical_pads_top(component=c, spacing=(-150, 30))
        cc.plot()

    """
    c = Component()
    component = gf.get_component(component)

    c.component = component
    ref = c << component
    ports_electrical = select_ports(ref.ports)
    ports_electrical = list(ports_electrical.values())

    if direction == "top":
        pads = c << gf.get_component(
            pad_array, columns=len(ports_electrical), rows=1, orientation=270
        )
    elif direction == "right":
        pads = c << gf.get_component(
            pad_array, columns=1, rows=len(ports_electrical), orientation=270
        )
    pads.x = ref.x + spacing[0]
    pads.ymin = ref.ymax + spacing[1]
    ports_pads = list(pads.ports.values())

    ports_pads = gf.routing.sort_ports.sort_ports_x(ports_pads)
    ports_component = gf.routing.sort_ports.sort_ports_x(ports_electrical)

    for p1, p2 in zip(ports_component, ports_pads):
        c << route_quad(p1, p2, layer=layer)

    c.add_ports(ref.ports)

    # remove electrical ports
    for port in ports_electrical:
        c.ports.pop(port.name)

    c.add_ports(pads.ports)
    c.copy_child_info(component)
    c.auto_rename_ports(prefix_electrical=f"elec-{component.name}-")
    return c


if __name__ == "__main__":
    # FIXME
    # c = demo_mzi()
    # c = demo_straight()
    # c.show(show_ports=True)
    import gdsfactory as gf

    c = gf.components.straight_heater_metal()
    # c = gf.components.mzi_phase_shifter_top_heater_metal()
    cc = gf.routing.add_electrical_pads_top(component=c, spacing=(-150, 30))
    cc.show(show_ports=True)
