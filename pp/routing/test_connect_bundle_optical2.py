import pp


if __name__ == "__main__":
    """ connect forward """
    """ this case needs some fixing """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = [
        w.ports["E1"],
        w.ports["E0"],
    ]
    ports2 = [
        d.ports["W1"],
        d.ports["W0"],
    ]

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=False)
    c.add(r)
    pp.show(c)
