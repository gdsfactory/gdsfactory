import uuid
import pp


def connect_electrical_shortest_path(port1, port2):
    """ needs some work

    does not work for some cases

    """
    c = pp.Component(name="zz_conn_{}".format(uuid.uuid4()))
    layer = port1.layer
    p1x0 = port1.endpoints[0][0]
    p1y0 = port1.endpoints[0][1]
    p1x1 = port1.endpoints[1][0]
    p1y1 = port1.endpoints[1][1]

    p2x0 = port2.endpoints[0][0]
    p2y0 = port2.endpoints[0][1]
    p2x1 = port2.endpoints[1][0]
    p2y1 = port2.endpoints[1][1]
    c.add_polygon(
        ([(p1x0, p1y0), (p1x1, p1y1), (p2x1, p2y1), (p2x0, p2y0)]), layer=layer
    )
    return c.ref()


if __name__ == "__main__":
    from pp.components.electrical.pad import pad

    c = pp.Component()
    p = pad()
    p1 = p.ref(position=(0, 0))
    p2 = p.ref(position=(200, 0))
    c.add(p1)
    c.add(p2)
    route = connect_electrical_shortest_path(port1=p1.ports["E"], port2=p2.ports["W"])
    c.add(route)
    pp.show(c)
