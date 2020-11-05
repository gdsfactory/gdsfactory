import pp


def demo_broken():
    """ shows error """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="E")

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=False)
    c.add(r)
    return c


if __name__ == "__main__":
    """ connects backwards """
    """ this case needs some fixing """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn()
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="E")

    ports1 = [
        w.ports["E0"],
        w.ports["E1"],
    ]
    ports2 = [
        d.ports["E1"],
        d.ports["E0"],
    ]

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True)
    c.add(r)
    pp.show(c)
