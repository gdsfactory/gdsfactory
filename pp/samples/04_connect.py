""" based on phidl tutorial

# Connecting devices with connect()

The connect command allows us to connect DeviceReference ports together like
Lego blocks.  There is an optional parameter called ``overlap`` which is
useful if you have shapes you want to intersect with some overlap (or with a
negative number, separate the ports).

"""


import pp

if __name__ == "__main__":
    c = pp.Component("waveguides_sample")

    wg1 = c << pp.c.waveguide(length=10, width=1)
    wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)
    wg3 = c << pp.c.waveguide(length=10, width=3, layer=pp.LAYER.SLAB150)

    wg2.connect(port="W0", destination=wg1.ports["E0"])
    wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)

    pp.show(c)  # show it in klayout
