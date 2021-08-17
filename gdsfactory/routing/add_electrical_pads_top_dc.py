from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.pad import pad_array as pad_array_function
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing.get_bundle import get_bundle
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.types import ComponentFactory


@cell
def add_electrical_pads_top_dc(
    component: Component,
    dy: float = 100.0,
    pad_array: ComponentFactory = pad_array_function,
    select_ports=select_ports_electrical,
    **kwargs,
) -> Component:
    """connects component electrical ports with pad array at the top

    Args:
        component:
        dy:
        pad_array:
        **kwargs: cross-section settings
    """
    c = Component()
    ports = select_ports(component.ports)
    ports = list(ports.values())
    for port in ports:
        port.orientation = 90

    c << component
    pads = c << pad_array(n=len(ports))
    pads.x = component.x
    pads.ymin = component.ymax + dy

    ports_pads = list(pads.ports.values())

    ports = sort_ports_x(ports)
    ports_pads = sort_ports_x(ports_pads)

    routes = get_bundle(ports, ports_pads, **kwargs)
    for route in routes:
        c.add(route.references)

    c.ports = component.ports.copy()
    for port in ports:
        c.ports.pop(port.name)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight_heater_metal(length=100.0)
    cc = add_electrical_pads_top_dc(component=c, layer=(31, 0), width=10)
    cc.show()
