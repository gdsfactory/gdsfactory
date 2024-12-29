import math

import gdsfactory as gf
from gdsfactory.components.bends.bend_circular import bend_circular


def test_add_padding_size() -> None:
    padding = 10
    c = bend_circular().copy()
    xsize_before = c.xsize
    dxsize_before = c.dxsize
    ysize_before = c.ysize
    dysize_before = c.dysize

    c = gf.add_padding(c, default=padding)

    assert c.xsize > xsize_before
    assert c.dxsize > dxsize_before
    assert c.ysize > ysize_before
    assert c.dysize > dysize_before

    assert math.isclose(c.xsize - xsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dxsize - dxsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.ysize - ysize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dysize - dysize_before, 2 * padding, abs_tol=0.01)


def test_add_padding_to_size() -> None:
    xsize = 10
    ysize = 10
    c = bend_circular().copy()
    xsize += c.xsize
    ysize += c.ysize

    c = gf.add_padding_to_size(c, xsize=xsize, ysize=ysize)

    assert math.isclose(c.xsize, xsize, abs_tol=0.01)
    assert math.isclose(c.dxsize, xsize, abs_tol=0.01)
    assert math.isclose(c.ysize, ysize, abs_tol=0.01)
    assert math.isclose(c.dysize, ysize, abs_tol=0.01)
