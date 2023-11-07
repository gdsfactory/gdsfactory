from __future__ import annotations

import gdsfactory as gf
from gdsfactory.functions import add_marker_layer
from gdsfactory.typings import Component

RANDOM_MARKER_LAYER = (111, 111)
RANDOM_MARKER_LABEL = "TEST_LABEL"


# @pytest.fixture(scope="function")
@gf.cell_with_child(cache=False)
def component(width=10, height=20) -> Component:
    c = gf.Component()
    c.add_polygon([(0, 0), (width, 0), (width, height), (0, height)], layer=(1, 0))
    return c


def test_add_marker_layer() -> None:
    my_component = add_marker_layer(marker_layer=RANDOM_MARKER_LAYER)(component)

    c = my_component()
    assert RANDOM_MARKER_LAYER in c.layers


def test_add_marker_layer_label_added() -> None:
    my_component = add_marker_layer(
        marker_layer=RANDOM_MARKER_LAYER, marker_label=RANDOM_MARKER_LABEL
    )(component)

    c = my_component()
    assert RANDOM_MARKER_LAYER in c.layers
    assert (label := c.labels[0]).text == RANDOM_MARKER_LABEL
    assert (label.layer, label.texttype) == RANDOM_MARKER_LAYER


def test_add_marker_layer_kwargs_passed() -> None:
    my_component = add_marker_layer(marker_layer=RANDOM_MARKER_LAYER)(component)

    c = my_component(height=50)
    assert RANDOM_MARKER_LAYER in c.layers
    assert c.size[1] == 50


if __name__ == "__main__":
    test_add_marker_layer()
