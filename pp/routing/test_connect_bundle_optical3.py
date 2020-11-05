import pp


if __name__ == "__main__":
    """ this case needs some fixing """
    """ connect 4 waveguides into a 4x1 component """
    c = pp.Component()

    w = c << pp.c.waveguide_array(n_waveguides=4, spacing=200)
    d = c << pp.c.nxn(west=4, east=1)
    d.y = w.y
    d.xmin = w.xmax + 200

    ports1 = w.get_ports_list(prefix="E")
    ports2 = d.get_ports_list(prefix="W")

    ports2.reverse()

    r = pp.routing.link_optical_ports(ports1, ports2, sort_ports=True)
    c.add(r)
    pp.show(c)
