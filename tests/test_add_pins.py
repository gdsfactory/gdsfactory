from __future__ import annotations

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_add_pins() -> None:
    """Ensure that all the waveguide has 2 pins."""
    cross_section = "strip"
    c = gf.components.straight(length=1.132, cross_section=cross_section)
    c = gf.add_pins.add_pins_container(c, layer=LAYER.PORT, layer_label=LAYER.TEXT)
    assert len(c.get_polygons()[LAYER.PORT]) == 2, len(c.get_polygons()[LAYER.PORT])


def test_add_pins_triangle() -> None:
    """Ensure that all the waveguide has 2 pins."""
    cross_section = "strip"
    c = gf.components.straight(length=1.139, cross_section=cross_section)

    add_pins_triangle = gf.partial(gf.add_pins.add_pins_triangle, layer=LAYER.PORT)

    c = gf.add_pins.add_pins_container(c, function=add_pins_triangle)
    assert len(c.get_polygons()[LAYER.PORT]) == 2, len(c.get_polygons()[LAYER.PORT])
