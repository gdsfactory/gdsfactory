import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.route_quad import route_quad
from gdsfactory.types import ComponentOrFactory


@gf.cell
def add_electrical_pads_shortest(
    component: Component,
    pad: ComponentOrFactory = pad_function,
    pad_port_spacing: float = 50.0,
    select_ports=select_ports_electrical,
    port_orientation: int = 90,
    layer: gf.types.Layer = (31, 0),
    **kwargs,
) -> Component:
    """Add pad to each closest electrical port.

    Args:
        component:
        pad: pad element or function
        pad_port_spacing: between pad and port
        select_ports: function
        port_orientation
        layer: for the routing
        **kwargs: pad_settings

    """
    c = Component()
    c.component = component
    ref = c << component
    ports = select_ports(ref.ports)
    ports = list(ports.values())

    pad = pad(**kwargs) if callable(pad) else pad
    pad_port_spacing += pad.info_child.full["size"][0] / 2

    for port in ports:
        p = c << pad
        if port_orientation == 0:
            p.x = port.x + pad_port_spacing
            p.y = port.y
            c.add(route_quad(port, p.ports["e1"], layer=layer))
        elif port_orientation == 180:
            p.x = port.x - pad_port_spacing
            p.y = port.y
            c.add(route_quad(port, p.ports["e3"], layer=layer))
        elif port_orientation == 90:
            p.y = port.y + pad_port_spacing
            p.x = port.x
            c.add(route_quad(port, p.ports["e4"], layer=layer))
        elif port_orientation == 270:
            p.y = port.y - pad_port_spacing
            p.x = port.x
            c.add(route_quad(port, p.ports["e2"], layer=layer))

    c.add_ports(ref.ports)
    for port in ports:
        c.ports.pop(port.name)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.cross(length=100, layer=gf.LAYER.M3)
    c = gf.components.mzi_phase_shifter()
    c = gf.components.straight_heater_metal()
    cc = add_electrical_pads_shortest(component=c)
    cc.show()
