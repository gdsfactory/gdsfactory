from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.functions import add_marker_layer
from gdsfactory.typings import Component

RANDOM_MARKER_LAYER = (111, 111)
RANDOM_MARKER_LABEL = "TEST_LABEL"

TEST_COMPONENT_LAYER_A = (1, 0)
TEST_COMPONENT_LAYER_B = (2, 0)


@gf.cell
def component(width=10, height=20) -> Component:
    c = gf.Component()
    c.add_polygon(
        [(0, 0), (width, 0), (width, height), (0, height)], layer=TEST_COMPONENT_LAYER_A
    )
    c.add_polygon(
        [(0, 0), (width, 0), (width, height / 2), (0, height / 2)],
        layer=TEST_COMPONENT_LAYER_B,
    )
    return c


def test_add_marker_layer() -> None:
    add_marker_layer_function = partial(
        add_marker_layer, marker_layer=RANDOM_MARKER_LAYER
    )
    c = gf.Component()
    _ = c << component()
    add_marker_layer_function(c)
    assert RANDOM_MARKER_LAYER in c.layers


def test_add_marker_layer_label_added() -> None:
    add_marker_layer_function = partial(
        add_marker_layer,
        marker_layer=RANDOM_MARKER_LAYER,
        marker_label=RANDOM_MARKER_LABEL,
    )
    c = gf.Component()
    _ = c << component()
    add_marker_layer_function(c)
    assert RANDOM_MARKER_LAYER in c.layers
    assert (label := c.labels[0]).text == RANDOM_MARKER_LABEL
    assert (label.layer, label.texttype) == RANDOM_MARKER_LAYER


def test_add_marker_layer_kwargs_passed() -> None:
    my_component = partial(
        component, decorator=partial(add_marker_layer, marker_layer=RANDOM_MARKER_LAYER)
    )

    c = my_component(height=50)
    assert RANDOM_MARKER_LAYER in c.layers
    assert c.size[1] == 50


def test_add_marker_layer_layers_to_mark_passed() -> None:
    my_component = partial(
        component,
        decorator=partial(
            add_marker_layer,
            marker_layer=RANDOM_MARKER_LAYER,
            layers_to_mark=[TEST_COMPONENT_LAYER_B],
        ),
    )

    c = my_component()
    assert RANDOM_MARKER_LAYER in c.layers
    assert c.get_polygons(
        by_spec=TEST_COMPONENT_LAYER_B, as_shapely=True
    ) == c.get_polygons(by_spec=RANDOM_MARKER_LAYER, as_shapely=True)


if __name__ == "__main__":
    test_add_marker_layer()
