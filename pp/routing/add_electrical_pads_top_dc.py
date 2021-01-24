from pp.component import Component
from pp.components.electrical.pad import pad_array
from pp.container import container
from pp.routing.connect import connect_elec_waypoints
from pp.routing.connect_bundle import connect_bundle


@container
def add_electrical_pads_top_dc(
    component: Component,
    component_top_to_pad_bottom_distance: float = 100.0,
    route_filter=connect_elec_waypoints,
    **kwargs,
) -> Component:
    """connects component electrical ports with pad array at the top

    Args:
        component:
        pad: pad element
        spacing: pad array (x, y) spacing
        width: pad width
        height: pad height
        layer: pad layer
    """
    c = Component(f"{component.name}_e")
    ports = component.get_ports_list(port_type="dc")
    c << component
    pads = c << pad_array(n=len(ports), port_list=["S"], **kwargs)
    pads.x = component.x
    pads.ymin = component.ymax + component_top_to_pad_bottom_distance

    ports_pads = list(pads.ports.values())

    ports_pads.sort(key=lambda p: p.x)
    ports.sort(key=lambda p: p.x)

    routes = connect_bundle(ports_pads, ports, route_filter=route_filter)
    for route in routes:
        c.add(route["references"])

    c.ports = component.ports.copy()
    for port in ports:
        c.ports.pop(port.name)
    return c


if __name__ == "__main__":
    import pp

    c = pp.c.mzi2x2(with_elec_connections=True)
    c = pp.c.wg_heater_connected()
    cc = add_electrical_pads_top_dc(c)
    pp.show(cc)
