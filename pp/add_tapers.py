import pp


@pp.ports.deco_rename_ports
def add_tapers(component, taper):
    """ returns tapered component """
    c = pp.Component(name=component.name + "_t")
    c.add_ref(component)

    for i, port in enumerate(component.ports.values()):
        taper_ref = c << pp.call_if_func(taper)
        taper_ref.connect(taper_ref.ports["2"].name, port)
        c.add_port(name="{}".format(i), port=taper_ref.ports["1"])
    return c


if __name__ == "__main__":
    c = pp.c.waveguide(width=2)
    t = pp.c.taper(width2=2)
    cc = add_tapers(c, t)
    pp.show(cc)
