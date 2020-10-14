from pp.component import Component
from pp.container import container
from pp.components.electrical.pad import pad_array
from pp.routing.connect_electrical import connect_electrical_shortest_path


@container
def add_electrical_pads_top(component, **kwargs):
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
    pads.y = component.ymax + 100
    ports_pads = list(pads.ports.values())
    for p1, p2 in zip(ports_pads, ports):
        c.add(connect_electrical_shortest_path(p1, p2))

    c.ports = component.ports
    for port in ports:
        c.ports.pop(port.name)
    return c


if __name__ == "__main__":
    import pp

    c = pp.c.mzi2x2(with_elec_connections=True)
    cc = add_electrical_pads_top(c)
