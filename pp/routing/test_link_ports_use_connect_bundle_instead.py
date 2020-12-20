import pp

if __name__ == "__main__":
    """ this case needs a fix
    link_ports does not work for this case
    use connect_bundle instead
    """

    w = h = 10
    c = pp.Component()
    pad_south = pp.c.pad_array(port_list=["S"], spacing=(15, 0), width=w, height=h)
    pt = c << pad_south
    pb = c << pad_south
    pb.rotate(90)
    pt.rotate(90)
    pb.move((0, -100))

    pbports = pb.get_ports_list()
    ptports = pt.get_ports_list()

    pbports.reverse()

    # routes = pp.routing.connect_bundle(pbports, ptports)
    routes = pp.routing.link_ports(pbports, ptports)
    c.add(routes)
    pp.show(c)
