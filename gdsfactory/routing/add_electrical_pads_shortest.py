import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.electrical.pad import pad
from gdsfactory.routing.get_route_electrical_shortest_path import (
    get_route_electrical_shortest_path,
)
from gdsfactory.types import ComponentOrFactory


@gf.cell
def add_electrical_pads_shortest(
    component: Component,
    pad: ComponentOrFactory = pad,
    pad_port_spacing: float = 50.0,
    **kwargs,
) -> Component:
    """Add pad to each closest electrical port.

    Args:
        component:
        pad: pad element or function
        pad_port_spacing: between pad and port
        width: pad width
        height: pad height
        layer: pad layer

    """
    c = Component(f"{component.name}_e")
    ports = component.get_ports_list(port_type="dc")
    c << component

    pad = pad(**kwargs) if callable(pad) else pad
    pad_port_spacing += pad.settings["width"] / 2

    for port in ports:
        p = c << pad
        if port.orientation == 0:
            p.x = port.x + pad_port_spacing
            p.y = port.y
            c.add(get_route_electrical_shortest_path(port, p.ports["W"]))
        elif port.orientation == 180:
            p.x = port.x - pad_port_spacing
            p.y = port.y
            c.add(get_route_electrical_shortest_path(port, p.ports["E"]))
        elif port.orientation == 90:
            p.y = port.y + pad_port_spacing
            p.x = port.x
            c.add(get_route_electrical_shortest_path(port, p.ports["S"]))
        elif port.orientation == 270:
            p.y = port.y - pad_port_spacing
            p.x = port.x
            c.add(get_route_electrical_shortest_path(port, p.ports["N"]))

    c.ports = component.ports.copy()
    for port in ports:
        c.ports.pop(port.name)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.cross(length=100, layer=gf.LAYER.M3, port_type="dc")
    c = gf.components.mzi_phase_shifter()
    c = gf.components.straight_with_heater()
    cc = add_electrical_pads_shortest(component=c)
    cc.show()
