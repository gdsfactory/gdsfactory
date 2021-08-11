from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.electrical.pad import pad_array as pad_array_function
from gdsfactory.routing.get_bundle import get_bundle
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.types import ComponentFactory


@cell
def add_electrical_pads_top_dc(
    component: Component,
    dy: float = 100.0,
    pad_array: ComponentFactory = pad_array_function,
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
    ports = component.get_ports_list(port_type="dc")
    c << component
    pads = c << pad_array(n=len(ports), port_list=("S",))
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

    c = gf.components.straight_with_heater(length=100.0)
    cc = add_electrical_pads_top_dc(component=c, layer=(31, 0), width=10)
    cc.show()
