"""
In this example we change the layer of the waveguide
"""

import pp


@pp.cell
def wg(layer=(2, 0), **kwargs):
    return pp.components.waveguide(layer=layer, **kwargs)


if __name__ == "__main__":
    c = wg()
    c.show()
