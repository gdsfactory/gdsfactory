"""
In this example we change the layer of the waveguide
"""

import pp


@pp.autoname
def wg(layer=(2, 0), **kwargs):
    return pp.c.waveguide(layer=layer, **kwargs)


if __name__ == "__main__":
    c = wg()
    pp.show(c)
