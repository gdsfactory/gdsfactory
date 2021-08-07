from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.routing.get_bundle import get_bundle
from gdsfactory.routing.sort_ports import sort_ports_x
from gdsfactory.tech import LIBRARY, Library
from gdsfactory.types import StrOrDict


@cell
def add_electrical_pads_top_dc(
    component: Component,
    dy: float = 100.0,
    library: Library = LIBRARY,
    pad_array: StrOrDict = "pad_array",
    **kwargs,
) -> Component:
    """connects component electrical ports with pad array at the top

    Args:
        component:
        dy:
        library

    """
    c = Component()
    ports = component.get_ports_list(port_type="dc")
    c << component
    pads = c << library.get_component(
        pad_array,
        n=len(ports),
        port_list=["S"],
    )
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

    c = gf.components.straight_with_heater()
    cc = add_electrical_pads_top_dc(component=c, waveguide="metal_routing")
    cc.show()
