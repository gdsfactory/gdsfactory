import uuid
from pp.component import Component
from pp.container import container
from pp.components.electrical.pad import pad_array, pad


def connect_electrical_shortest_path(port1, port2):
    """connects two ports with a polygon that takes the shortest path"""
    c = Component(name="zz_conn_{}".format(uuid.uuid4()))
    layer = port1.layer
    p1x0 = port1.endpoints[0][0]
    p1y0 = port1.endpoints[0][1]
    p1x1 = port1.endpoints[1][0]
    p1y1 = port1.endpoints[1][1]

    p2x0 = port2.endpoints[0][0]
    p2y0 = port2.endpoints[0][1]
    p2x1 = port2.endpoints[1][0]
    p2y1 = port2.endpoints[1][1]

    if port1.orientation in [90, 270]:
        c.add_polygon(
            ([(p1x1, p1y0), (p1x0, p1y1), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
        )
    else:
        c.add_polygon(
            ([(p1x0, p1y1), (p1x1, p1y0), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
        )
    return c.ref()


@container
def connect_electrical_pads_top(component, **kwargs):
    """connects component electrical ports with pad array at the top

    Args:
        component:
        pad: pad element
        spacing: pad array (x, y) spacing
        width: pad width
        height: pad height
        layer: pad layer
    """
    c = Component(f"{component}_e")
    ports = component.get_electrical_ports()
    c << component
    pads = c << pad_array(n=len(ports), port_list=["S"], **kwargs)
    pads.x = component.x
    pads.y = component.ymax + 100
    ports_pads = list(pads.ports.values())
    for p1, p2 in zip(ports_pads, ports):
        c.add(connect_electrical_shortest_path(p1, p2))
    return c


@container
def connect_electrical_pads_shortest(component, pad=pad, pad_port_spacing=50, **kwargs):
    """add a pad to each closest electrical port
    Args:
        component:
        pad: pad element or function
        pad_port_spacing: between pad and port
        width: pad width
        height: pad height
        layer: pad layer

    """
    c = Component(f"{component}_e")
    ports = component.get_electrical_ports()
    c << component

    pad = pad(**kwargs) if callable(pad) else pad
    pad_port_spacing += pad.settings["width"] / 2

    for port in ports:
        p = c << pad
        if port.orientation == 0:
            p.movex(port.x + pad_port_spacing)
            p.y = port.y
            c.add(connect_electrical_shortest_path(port, p.ports["W"]))
        elif port.orientation == 180:
            p.movex(port.x - pad_port_spacing)
            p.y = port.y
            c.add(connect_electrical_shortest_path(port, p.ports["E"]))
        elif port.orientation == 90:
            p.movey(port.y + pad_port_spacing)
            p.x = port.x
            c.add(connect_electrical_shortest_path(port, p.ports["S"]))
        elif port.orientation == 270:
            p.movey(port.y - pad_port_spacing)
            p.x = port.x
            c.add(connect_electrical_shortest_path(port, p.ports["N"]))

    return c


def demo():
    import pp

    c = Component()
    p = pad()
    p1 = p.ref(position=(0, 0))
    p2 = p.ref(position=(200, 0))
    c.add(p1)
    c.add(p2)
    route = connect_electrical_shortest_path(port1=p1.ports["E"], port2=p2.ports["W"])
    c.add(route)
    pp.show(c)


def demo2():
    import pp

    c = Component("mzi_with_pads")
    mzi = pp.c.mzi2x2(with_elec_connections=True)
    pads = pad_array(n=3, port_list=["S"])
    p = c << pads
    c << mzi
    p.move((-150, 200))
    ports_pads = list(p.ports.values())
    ports_mzi = mzi.get_electrical_ports()

    for p1, p2 in zip(ports_pads, ports_mzi):
        c.add(connect_electrical_shortest_path(p1, p2))
    pp.show(c)


if __name__ == "__main__":
    import pp

    # c = pp.c.mzi2x2(with_elec_connections=True)
    # cc = connect_electrical_pads_top(c)

    c = pp.c.cross(length=100, layer=pp.LAYER.M3, port_type="dc")
    c.move((20, 50))
    cc = connect_electrical_pads_shortest(c)
    pp.show(cc)
