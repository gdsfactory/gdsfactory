""" based on phidl tutorial

# Connecting devices with connect()

The connect command allows us to connect DeviceReference ports together like
Lego blocks.  There is an optional parameter called ``overlap`` which is
useful if you have shapes you want to intersect with some overlap (or with a
negative number, separate the ports).

"""


import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("straights_sample")

    wg1 = c << gf.components.straight(length=10, width=1)
    wg2 = c << gf.components.straight(length=10, width=2, layer=gf.LAYER.SLAB90)
    wg3 = c << gf.components.straight(length=10, width=3, layer=gf.LAYER.SLAB150)

    wg2.connect(port="o1", destination=wg1.ports["o2"])
    wg3.connect(port="o1", destination=wg2.ports["o2"], overlap=1)

    c.show()  # show it in klayout
