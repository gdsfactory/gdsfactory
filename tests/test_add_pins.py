from __future__ import annotations

from functools import partial

import pytest

import gdsfactory as gf
from gdsfactory.add_pins import (
    add_bbox_siepic,
    add_pin_rectangle_inside,
    add_pins_siepic,
)
from gdsfactory.component import Component
from gdsfactory.generic_tech import LAYER
from gdsfactory.port import Port

cladding_layers_optical_siepic = ("DEVREC",)  # for SiEPIC verification
cladding_offsets_optical_siepic = (0,)  # for SiEPIC verification

add_pins_siepic_100nm = partial(add_pins_siepic, pin_length=0.1)

strip_siepic100nm = partial(
    gf.cross_section.cross_section,
    add_pins=add_pins_siepic_100nm,
    add_bbox=add_bbox_siepic,
    cladding_layers=cladding_layers_optical_siepic,
    cladding_offsets=cladding_offsets_optical_siepic,
)


@pytest.mark.parametrize("optical_routing_type", [0, 1])
def test_add_pins_with_routes(optical_routing_type) -> None:
    """Add pins to a straight ensure that all the routes have pins."""
    cross_section = "xs_sc_pins"
    c = gf.components.straight(length=1.0, cross_section=cross_section)
    gc = gf.components.grating_coupler_elliptical_te(cross_section=cross_section)
    cc = gf.routing.add_fiber_single(
        component=c,
        grating_coupler=[gc, gf.components.grating_coupler_tm],
        optical_routing_type=optical_routing_type,
        cross_section=cross_section,
    )
    pins_component = cc.extract(layers=(LAYER.PORT,))
    assert len(pins_component.polygons) == 12, len(pins_component.polygons)


def test_add_pins() -> None:
    """Ensure that all the waveguide has 2 pins."""
    cross_section = "xs_sc_pins"
    c = gf.components.straight(length=1.0, cross_section=cross_section)
    pins_component = c.extract(layers=(LAYER.PORT,))
    assert len(pins_component.polygons) == 2, len(pins_component.polygons)


def test_add_pin_rectangle_inside() -> None:
    """Test that a square pin is added towards the inside of the port."""
    c = Component()
    w = 1.0
    port = Port(
        name="test_port",
        center=(0, 0),
        width=w,
        orientation=0,
        layer=(1, 0),
    )
    c.add_port(port)
    add_pin_rectangle_inside(
        component=c,
        port=port,
        pin_length=0.1,
        layer=(2, 0),
        layer_label=(3, 0),
        label_function=None,
    )
    assert len(c.polygons) == 1, len(c.polygons)
    assert len(c.labels) == 1, len(c.labels)
    assert c.labels[0].text == "test_port", c.labels[0].text
    assert c.labels[0].origin == (0, 0), c.labels[0].origin
    assert (c.labels[0].layer, c.labels[0].texttype) == (3, 0), (
        c.labels[0].layer,
        c.labels[0].texttype,
    )


def test_add_pin_rectangle_inside_with_label_function() -> None:
    """Test that a square pin is added towards the inside of the port with a custom label."""
    c = Component(name="test_add_pins")
    w = 1.0
    port = Port(
        name="test_port",
        center=(0, 0),
        width=w,
        orientation=0,
        layer=(1, 0),
    )
    c.add_port(port)

    def label_function(
        component: Component, rough_component_name: str, port: Port
    ) -> str:
        assert (
            rough_component_name == "test_add_pin_rectangle_inside_with_label_function"
        ), rough_component_name
        return f"{component.name}_{port.name}_test"

    add_pin_rectangle_inside(
        component=c,
        port=port,
        pin_length=0.1,
        layer=(2, 0),
        layer_label=(3, 0),
        label_function=label_function,
    )
    assert len(c.polygons) == 1, len(c.polygons)
    assert len(c.labels) == 1, len(c.labels)
    assert c.labels[0].text == "test_add_pins_test_port_test", c.labels[0].text
    assert c.labels[0].origin == (0, 0), c.labels[0].origin
    assert (c.labels[0].layer, c.labels[0].texttype) == (3, 0), (
        c.labels[0].layer,
        c.labels[0].texttype,
    )


if __name__ == "__main__":
    # test_add_pins()
    # test_add_pins_with_routes(0)
    # test_add_pin_rectangle_inside()
    test_add_pin_rectangle_inside_with_label_function()
