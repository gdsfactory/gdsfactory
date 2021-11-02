"""Trying to replicate a duplicated cell error coming from extruding a path.

"""

import gdsfactory as gf


def wg(size=(1, 0.5), layer=(1, 0)):
    """Dummy component"""
    c = gf.Component("wg")
    dx, dy = size

    points = [
        [-dx / 2.0, -dy / 2.0],
        [-dx / 2.0, dy / 2],
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2.0],
    ]

    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = gf.Component()
    s1 = c << wg()
    s2 = c << wg()
    c.show()
