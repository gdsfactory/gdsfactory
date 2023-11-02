from __future__ import annotations

import gdsfactory as gf

r1 = (8, 8)
r2 = (11, 4)

angle_resolution = 2.5

c1 = gf.components.ellipse(radii=r1, layer=(1, 0), angle_resolution=angle_resolution)
c2 = gf.components.ellipse(radii=r2, layer=(1, 0), angle_resolution=angle_resolution)


def test_boolean_not() -> None:
    c4 = gf.boolean(c1, c2, operation="not", layer=(1, 0))
    assert int(c4.area()) == 87


def test_boolean_or() -> None:
    c4 = gf.boolean(c1, c2, operation="or", layer=(1, 0))
    assert int(c4.area()) == 225


def test_boolean_xor() -> None:
    c4 = gf.boolean(c1, c2, operation="xor", layer=(1, 0))
    assert int(c4.area()) == 111


def test_boolean_and() -> None:
    c4 = gf.boolean(c1, c2, operation="and", layer=(1, 0))
    assert int(c4.area()) == 113
