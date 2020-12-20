""" From phidl tutorial

# Flattening a Component

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.  The c.flatten() keeps all the
polygons in c, but removes all the underlying references it's attached to.
Also, if you specify the `single_layer` argument it will move all of the
polyons to that single layer.

"""


if __name__ == "__main__":
    import pp

    c = pp.Component("waveguides_sample")

    wg1 = c << pp.c.waveguide(length=10, width=1)
    wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)
    wg3 = c << pp.c.waveguide(length=10, width=3, layer=pp.LAYER.SLAB150)

    wg2.connect(port="W0", destination=wg1.ports["E0"])
    wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)
    c.flatten()

    pp.show(c)  # show it in klayout
