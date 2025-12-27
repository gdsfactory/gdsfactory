from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import LAYER


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


def test_add_bbox() -> None:
    c = gf.Component()
    layer = LAYER.DEVREC
    c = gf.add_pins.add_bbox(
        component=c,
        bbox_layer=layer,
        top=0.5,
        bottom=0.5,
        left=0.5,
        right=0.5,
    )
    bbox = c.bbox_np()
    assert bbox[0, 0] == -0.5
    assert bbox[1, 0] == 0.5
    assert bbox[0, 1] == -0.5
    assert bbox[1, 1] == 0.5


def test_add_pins_siepic() -> None:
    c = gf.components.straight(length=10).copy()
    c = gf.add_pins.add_pins_siepic(c)
    assert len(c.get_polygons()[LAYER.PORT]) == 2


def test_add_pins_siepic_electrical() -> None:
    c = gf.components.straight_heater_metal().copy()
    c = gf.add_pins.add_pins_siepic_electrical(c)
    assert len(c.get_polygons()[LAYER.PORTE]) == 8


def test_add_outline() -> None:
    c = gf.components.straight().copy()
    gf.add_pins.add_outline(c, layer=LAYER.DEVREC)
    assert len(c.get_polygons()[LAYER.DEVREC]) == 1


def test_add_settings_label() -> None:
    c = gf.components.straight().copy()
    gf.add_pins.add_settings_label(c)
    assert len(c.get_labels(LAYER.LABEL_SETTINGS)) == 1


def test_add_instance_label() -> None:
    c = gf.Component()
    ref = c << gf.components.straight()
    gf.add_pins.add_instance_label(c, ref)
    assert len(c.get_labels(LAYER.LABEL_INSTANCE)) == 1


def test_add_pins_and_outline() -> None:
    c = gf.components.straight().copy()
    gf.add_pins.add_pins_and_outline(c)
    assert len(c.get_polygons()[LAYER.PORT]) == 2
    assert len(c.get_polygons()[LAYER.DEVREC]) == 1
