import klayout.db as pya


def check_inclusion(
    gdspath,
    layer_in=(1, 0),
    layer_out=(2, 0),
    min_inclusion=0.150,
    dbu=1e3,
    ignore_angle_deg=80,
    whole_edges=False,
    metrics=None,
    min_projection=None,
    max_projection=None,
):
    """reads layer from top cell and returns a the area that violates min inclusion
    if 0 no area violates exclusion

    Args:
        gdspath: path to GDS
        layer_in: tuple
        layer_out: tuple
        min_inclusion: in um
        dbu: database units (1000 um/nm)
        ignore_angle_deg: The angle above which no check is performed
        other: The other region against which to check
        whole_edges: If true, deliver the whole edges
        metrics: Specify the metrics type
        min_projection The lower threshold of the projected length of one edge onto another
        max_projection The upper limit of the projected length of one edge onto another
    """
    from pp.component import Component
    from pp.write_component import write_gds

    if isinstance(gdspath, Component):
        gdspath.flatten()
        gdspath = write_gds(gdspath)
    layout = pya.Layout()
    layout.read(str(gdspath))
    cell = layout.top_cell()
    a = pya.Region(cell.begin_shapes_rec(layout.layer(layer_in[0], layer_in[1])))
    b = pya.Region(cell.begin_shapes_rec(layout.layer(layer_out[0], layer_out[1])))

    d = b.inside_check(
        a,
        min_inclusion * dbu,
        whole_edges,
        metrics,
        ignore_angle_deg,
        min_projection,
        max_projection,
    )
    return d.polygons().area()


if __name__ == "__main__":
    import pp

    w1 = 0.5
    inclusion = 0.1
    w2 = w1 - inclusion
    min_inclusion = 0.11
    # min_inclusion = 0.01
    dbu = 1000
    layer = (1, 0)
    c = pp.Component()
    r1 = c << pp.c.rectangle(size=(w1, w1), layer=(1, 0))
    r2 = c << pp.c.rectangle(size=(w2, w2), layer=(2, 0))
    r1.x = 0
    r1.y = 0
    r2.x = 0
    r2.y = 0
    gdspath = c
    pp.show(gdspath)
    print(check_inclusion(c, min_inclusion=min_inclusion))

    # if isinstance(gdspath, pp.Component):
    #     gdspath.flatten()
    #     gdspath = pp.write_gds(gdspath)
    # layout = pya.Layout()
    # layout.read(str(gdspath))
    # cell = layout.top_cell()
    # region = pya.Region(cell.begin_shapes_rec(layout.layer(layer[0], layer[1])))

    # d = region.space_check(min_inclusion * dbu)
    # print(d.polygons().area())
