from __future__ import annotations

from collections.abc import Callable

import pytest

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


def test_add_instance_label() -> None:
    c = gf.Component()
    ref = c << gf.components.straight()
    gf.add_pins.add_instance_label(c, ref)
    assert len(c.get_labels(LAYER.LABEL_INSTANCE)) == 1


def test_add_instance_label_missing_layer() -> None:
    import gdsfactory as gf
    from gdsfactory.typings import LayerSpec

    original_pdk = gf.get_active_pdk()
    pdk = original_pdk.model_copy()
    pdk.layers = None
    pdk.activate()

    try:
        c = gf.Component("parent")
        ref = c << gf.Component("child")
        with pytest.warns(UserWarning, match="Layer LABEL_INSTANCE not found"):
            gf.add_pins.add_instance_label(c, ref)

        fallback_layer: LayerSpec = (1, 0)
        assert len(c.get_labels(fallback_layer)) == 1

    finally:
        original_pdk.activate()


def test_add_bbox_siepic() -> None:
    c = gf.components.mzi().copy()
    c = gf.add_pins.add_bbox_siepic(c, remove_layers=[])
    assert len(c.get_polygons()[LAYER.DEVREC]) == 1


def test_get_pin_triangle_polygon_tip() -> None:
    # Test without port_face
    port = gf.Port(
        name="o1",
        center=(0, 0),
        orientation=0,
        width=0.5,
        layer=LAYER.PORT,
    )
    polygon, tip = gf.add_pins.get_pin_triangle_polygon_tip(port)
    assert len(polygon) == 3
    assert polygon[0][1] == -0.25
    assert polygon[1][1] == 0.25
    assert tip[0] == 0.25
    assert tip[1] == 0

    # Test with port_face
    port_with_face = gf.Port(
        name="o2",
        center=(0, 0),
        orientation=0,
        width=0.5,
        layer=LAYER.PORT,
        info={"face": [(0, -0.3), (0, 0.3)]},  # type: ignore
    )
    polygon2, tip2 = gf.add_pins.get_pin_triangle_polygon_tip(port_with_face)
    assert len(polygon2) == 3
    assert polygon2[0][1] == 0.3
    assert polygon2[1][1] == -0.3
    assert tip2[0] == 0.25
    assert tip2[1] == 0


def test_add_pin_triangle() -> None:
    c = gf.Component()
    port = gf.Port(
        name="o1",
        center=(0, 0),
        orientation=0,
        width=0.5,
        layer=LAYER.PORT,
    )
    gf.add_pins.add_pin_triangle(
        component=c, port=port, layer=LAYER.PORT, layer_label=LAYER.TEXT
    )
    assert len(c.get_polygons()[LAYER.PORT]) == 1
    assert len(c.get_labels(LAYER.TEXT)) == 1

    c2 = gf.Component()
    gf.add_pins.add_pin_triangle(
        component=c2, port=port, layer=LAYER.PORT, layer_label=None
    )
    assert len(c2.get_polygons()[LAYER.PORT]) == 1
    assert len(c2.get_labels(LAYER.TEXT)) == 0


@pytest.mark.parametrize(
    "add_pin_function",
    [gf.add_pins.add_pin_rectangle_inside, gf.add_pins.add_pin_rectangle],
)
def test_add_pin_rectangle(
    add_pin_function: Callable[..., None],
) -> None:
    c = gf.Component()
    port = gf.Port(
        name="o1",
        center=(0, 0),
        orientation=0,
        width=0.5,
        layer=LAYER.PORT,
    )

    add_pin_function(
        component=c, port=port, pin_length=0.1, layer=LAYER.PORT, layer_label=LAYER.TEXT
    )
    assert len(c.get_polygons()[LAYER.PORT]) == 1
    assert len(c.get_labels(LAYER.TEXT)) == 1

    c2 = gf.Component()
    add_pin_function(
        component=c2, port=port, pin_length=0.1, layer=LAYER.PORT, layer_label=None
    )
    assert len(c2.get_polygons()[LAYER.PORT]) == 1
    assert len(c2.get_labels(LAYER.TEXT)) == 0

    c3 = gf.Component()
    add_pin_function(
        component=c3, port=port, pin_length=0.1, layer=None, layer_label=LAYER.TEXT
    )
    assert len(c3.get_polygons()) == 0
    assert len(c3.get_labels(LAYER.TEXT)) == 1

    c4 = gf.Component()
    add_pin_function(
        component=c4, port=port, pin_length=0.1, layer=None, layer_label=None
    )
    assert len(c4.get_polygons()) == 0
    assert len(c4.get_labels(LAYER.TEXT)) == 0


def test_add_settings_label() -> None:
    c = gf.components.straight().copy()
    gf.add_pins.add_settings_label(c)
    assert len(c.get_labels(LAYER.LABEL_SETTINGS)) == 1

    c = gf.Component()
    c.info.update({"test_key": "test_value"})

    gf.add_pins.add_settings_label(component=c)
    assert len(c.get_labels("LABEL_SETTINGS")) == 1
    label = c.get_labels("LABEL_SETTINGS")[0]
    assert "test_key" in label.string
    assert "test_value" in label.string

    c2 = gf.Component()
    c2.info.update({"test_key": "test_value"})
    gf.add_pins.add_settings_label(component=c2, with_yaml_format=True)
    assert len(c2.get_labels("LABEL_SETTINGS")) == 1
    label = c2.get_labels("LABEL_SETTINGS")[0]
    assert "test_key" in label.string
    assert "test_value" in label.string

    c = gf.Component()
    c.info.update({"test_key": "test_value"})

    c3 = gf.Component()
    ref = c3 << c
    gf.add_pins.add_settings_label(component=c3, reference=ref)
    c3.show()
    assert len(c3.get_labels("LABEL_SETTINGS")) == 1

    c4 = gf.Component()
    c4.info.update({"test_key": "test_value"})
    gf.add_pins.add_settings_label(component=c4, layer_label="TEXT")
    assert len(c4.get_labels("TEXT")) == 1

    c5 = gf.Component()
    c5.info.update({"long_key": "x" * 2000})
    with pytest.raises(ValueError, match="label > 1024 characters"):
        gf.add_pins.add_settings_label(component=c5)


def test_add_pins_and_outline_with_functions() -> None:
    c = gf.components.straight().copy()
    gf.add_pins.add_pins_and_outline(c)
    assert len(c.get_polygons()[LAYER.PORT]) == 2
    assert len(c.get_polygons()[LAYER.DEVREC]) == 1

    c = gf.components.straight().copy()
    gf.add_pins.add_pins_and_outline(c)
    assert len(c.get_polygons()[LAYER.PORT]) == 2
    assert len(c.get_polygons()[LAYER.DEVREC]) == 1
    assert len(c.get_labels(LAYER.LABEL_SETTINGS)) == 2

    c2 = gf.components.straight().copy()
    gf.add_pins.add_pins_and_outline(
        c2,
        add_outline_function=None,
        add_settings_function=None,
        add_instance_label_function=None,
    )
    assert len(c2.get_polygons()[LAYER.PORT]) == 2
    assert len(c2.get_labels(LAYER.LABEL_SETTINGS)) == 0

    c3 = gf.Component()
    ref = c3 << gf.components.straight()
    gf.add_pins.add_pins_and_outline(c3, reference=ref)
    assert len(c3.get_labels(LAYER.LABEL_SETTINGS)) == 2


if __name__ == "__main__":
    test_add_pins_and_outline_with_functions()
