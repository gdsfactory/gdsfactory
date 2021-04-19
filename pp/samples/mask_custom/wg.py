"""
In this example we change the layer of the straight
"""

import pp


@pp.cell
def wg(layer=(2, 0), **kwargs):
    return pp.components.straight(layer=layer, **kwargs)


if __name__ == "__main__":
    c = wg()
    c.show()
