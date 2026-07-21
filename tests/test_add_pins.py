from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.gpdk import LAYER


@pytest.fixture(autouse=True)
def activate_generic_pdk():
    gf.gpdk.PDK.activate()


def test_add_electric_pins_without_layer_map() -> None:
    """Pin rectangles are drawn on the port's own layer when no layer_map is given."""
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer=LAYER.M1)
    component.add_port(
        name="A",
        center=(0, 2.5),
        width=5,
        orientation=180,
        layer=LAYER.M1,
        port_type="electrical",
    )
    component.add_port(
        name="B",
        center=(10, 2.5),
        width=5,
        orientation=0,
        layer=LAYER.M1,
        port_type="electrical",
    )
    gf.add_pins.add_electric_pins(component)
    polygons = component.get_polygons()
    assert len(polygons[LAYER.M1]) == 3  # original polygon + 2 pin rectangles
    assert len(component.pins) == 2


def test_add_electric_pins_with_layer_map() -> None:
    """Pin rectangles are drawn on the mapped pin layer, not the port layer."""
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer=LAYER.M1)
    component.add_port(
        name="A",
        center=(0, 2.5),
        width=5,
        orientation=180,
        layer=LAYER.M1,
        port_type="electrical",
    )
    component.add_port(
        name="B",
        center=(10, 2.5),
        width=5,
        orientation=0,
        layer=LAYER.M1,
        port_type="electrical",
    )
    gf.add_pins.add_electric_pins(
        component, layer_map={LAYER.M1: LAYER.PORTE}
    )
    polygons = component.get_polygons()
    assert len(polygons[LAYER.PORTE]) == 2
    assert len(polygons[LAYER.M1]) == 1  # only the original polygon
    assert len(component.pins) == 2


def test_add_electric_pins_groups_ports_by_name() -> None:
    """Ports with the same name are grouped into a single logical pin."""
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1)
    component.add_port(
        name="D",
        center=(0, 2.5),
        width=5,
        orientation=180,
        layer=LAYER.M1,
        port_type="electrical",
    )
    component.add_port(
        name="D",
        center=(0, 7.5),
        width=5,
        orientation=180,
        layer=LAYER.M1,
        port_type="electrical",
    )
    component.add_port(
        name="S",
        center=(10, 5),
        width=10,
        orientation=0,
        layer=LAYER.M1,
        port_type="electrical",
    )
    gf.add_pins.add_electric_pins(component)
    pin_names = {pin.name for pin in component.pins}
    assert pin_names == {"D", "S"}
    assert len(component.pins) == 2


def test_add_electric_pins_skips_non_electrical_ports() -> None:
    """Only electrical ports get pins; optical ports are ignored."""
    component = gf.Component()
    component.add_polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)], layer=LAYER.WG)
    component.add_port(
        name="o1",
        center=(0, 0.25),
        width=0.5,
        orientation=180,
        layer=LAYER.WG,
        port_type="optical",
    )
    component.add_port(
        name="e1",
        center=(10, 0.25),
        width=0.5,
        orientation=0,
        layer=LAYER.M1,
        port_type="electrical",
    )
    gf.add_pins.add_electric_pins(component)
    assert len(component.pins) == 1
    assert component.pins[0].name == "e1"


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
