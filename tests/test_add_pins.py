from __future__ import annotations

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_add_pins() -> None:
    """Ensure that all the waveguide has 2 pins."""
    cross_section = "xs_sc"
    c = gf.components.straight(length=1.0, cross_section=cross_section)
    c = gf.add_pins.add_pins_container(c)
    pins_component = c.extract(layers=(LAYER.PORT,))
    assert len(pins_component.get_polygons()[(1, 10)]) == 3, len(
        pins_component.polygons
    )
