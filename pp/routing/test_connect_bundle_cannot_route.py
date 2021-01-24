import pp

if __name__ == "__main__":
    """FIXME: this case needs to be implemented for connect_bundle."""

    w = h = 10
    c = pp.Component()
    pad_south = pp.c.pad_array(port_list=["S"], spacing=(15, 0), width=w, height=h)
    pl = c << pad_south
    pb = c << pad_south
    pl.rotate(90)
    pb.move((100, -100))

    pbports = pb.get_ports_list()
    ptports = pl.get_ports_list()

    routes = pp.routing.connect_bundle(pbports, ptports)
    for route in routes:
        c.add(route["references"])
    pp.show(c)
