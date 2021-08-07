"""
In this example we change the layer of the straight
"""

import gdsfactory as gf


@gf.cell
def wg(layer=(2, 0), **kwargs):
    return gf.components.straight(layer=layer, **kwargs)


if __name__ == "__main__":
    c = wg()
    c.show()
