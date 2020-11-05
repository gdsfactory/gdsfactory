import pp
from pp.port import deco_rename_ports


@deco_rename_ports
@pp.autoname
def floorplan(
    inputs=[(0, 4, 180)],
    outputs=[(8, 2, 0), (8, 6, 0)],
    xsize=8,
    ysize=8,
    wg_width=0.5,
    layer=pp.LAYER.WG,
):
    """ returns the floorplan of a component """
    c = pp.Component()
    c << pp.c.rectangle(size=(xsize, ysize), layer=layer)
    i = 0
    for ii in inputs:
        x, y, orientation = ii
        c.add_port(
            i, midpoint=(x, y), width=wg_width, orientation=orientation, layer=layer
        )
        i += 1
    for oo in outputs:
        x, y, orientation = oo
        c.add_port(
            i, midpoint=(x, y), width=wg_width, orientation=orientation, layer=layer
        )
        i += 1
    return c


if __name__ == "__main__":
    c = floorplan()
    pp.show(c)
