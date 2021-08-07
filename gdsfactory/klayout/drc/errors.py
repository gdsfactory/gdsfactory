import gdsfactory

layer = gdsfactory.LAYER.WG


@gdsfactory.cell
def width_min(size=(0.1, 0.1)):
    c = gdsfactory.components.rectangle(size=size, layer=layer)
    return c


@gdsfactory.cell
def gap_min(gap=0.1):
    c = gdsfactory.Component()
    r1 = c << gdsfactory.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gdsfactory.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gdsfactory.cell
def snapping_error(gap=1e-3):
    c = gdsfactory.Component()
    r1 = c << gdsfactory.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gdsfactory.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gdsfactory.cell
def errors():
    D_list = [width_min(), gap_min()]
    c = gdsfactory.pack(D_list, spacing=1.5)
    return c[0]


if __name__ == "__main__":
    # c = width_min()
    # c.write_gds("wmin.gds")
    # c = gap_min()
    # c.write_gds("gmin.gds")
    # c = snapping_error()
    # c.write_gds("snap.gds")

    c = errors()
    c.write_gds("errors.gds")
    c.show()
