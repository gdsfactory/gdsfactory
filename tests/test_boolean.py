from __future__ import annotations

import pytest

import gdsfactory as gf

layer = (1, 0)
r1 = (8, 8)
r2 = (11, 4)

angle_resolution = 2.5


@pytest.fixture
def c1() -> gf.Component:
    return gf.components.ellipse(
        radii=r1, layer=(1, 0), angle_resolution=angle_resolution
    )


@pytest.fixture
def c2() -> gf.Component:
    return gf.components.ellipse(
        radii=r2, layer=(1, 0), angle_resolution=angle_resolution
    )


def test_boolean_not(c1: gf.Component, c2: gf.Component) -> None:
    c4 = gf.boolean(c1, c2, operation="not", layer=layer)
    assert int(c4.area(layer=layer)) == 87


def test_boolean_or(c1: gf.Component, c2: gf.Component) -> None:
    c4 = gf.boolean(c1, c2, operation="or", layer=layer)
    assert int(c4.area(layer=layer)) == 225


def test_boolean_xor(c1: gf.Component, c2: gf.Component) -> None:
    c4 = gf.boolean(c1, c2, operation="xor", layer=layer)
    assert int(c4.area(layer=layer)) == 111


def test_boolean_and(c1: gf.Component, c2: gf.Component) -> None:
    c4 = gf.boolean(c1, c2, operation="and", layer=layer)
    assert int(c4.area(layer=layer)) == 113
