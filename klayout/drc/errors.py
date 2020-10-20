import pp

layer = pp.LAYER.WG


@pp.autoname
def width_min(size=(0.1, 0.1)):
    c = pp.c.rectangle(size=size, layer=layer)
    return c


@pp.autoname
def gap_min(gap=0.1):
    c = pp.Component()
    r1 = c << pp.c.rectangle(size=(1, 1), layer=layer)
    r2 = c << pp.c.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@pp.autoname
def snapping_error(gap=1e-3):
    c = pp.Component()
    r1 = c << pp.c.rectangle(size=(1, 1), layer=layer)
    r2 = c << pp.c.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@pp.autoname
def errors():
    D_list = [width_min(), gap_min()]
    c = pp.pack(D_list, spacing=1.5)
    return c[0]


if __name__ == "__main__":
    # c = width_min()
    # pp.write_gds(c, "wmin.gds")
    # c = gap_min()
    # pp.write_gds(c, "gmin.gds")
    # c = snapping_error()
    # pp.write_gds(c, "snap.gds")

    c = errors()
    pp.write_gds(c, "errors.gds")
    pp.show(c)
