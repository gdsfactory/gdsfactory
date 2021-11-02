"""This does not fail

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
    c1 = gf.Component("wg1")
    s1 = c1 << wg()

    c2 = gf.Component("wg2")
    s2 = c2 << wg()

    c = gf.Component("TOP")
    c << c1
    c << c2
    c.write_gds("a.gds")
    c.show()
