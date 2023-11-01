from __future__ import annotations

import gdsfactory as gf
from gdsfactory.functions import add_marker_layer
from gdsfactory.typings import Component

RANDOM_MARKER_LAYER = (111, 111)


def test_add_marker_layer() -> None:
    @add_marker_layer(marker_layer=RANDOM_MARKER_LAYER)
    @gf.cell_with_child
    def my_component(width=10, height=20) -> Component:
        c = gf.Component()
        c.add_polygon([(0, 0), (width, 0), (width, height), (0, height)], layer=(1, 0))
        return c

    c = my_component()
    assert RANDOM_MARKER_LAYER in c.layers


def test_add_marker_layer_kwargs_passed() -> None:
    @add_marker_layer(marker_layer=RANDOM_MARKER_LAYER)
    @gf.cell_with_child
    def my_component(width=10, height=20) -> Component:
        c = gf.Component()
        c.add_polygon([(0, 0), (width, 0), (width, height), (0, height)], layer=(1, 0))
        return c

    c = my_component(height=50)
    assert RANDOM_MARKER_LAYER in c.layers
    assert c.size[1] == 50


if __name__ == "__main__":
    test_add_marker_layer()
