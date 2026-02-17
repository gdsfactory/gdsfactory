from __future__ import annotations

import math
from functools import partial

import pytest

import gdsfactory as gf
from gdsfactory.add_ports import (
    add_ports_from_labels,
    add_ports_from_markers_inside,
    add_ports_from_siepic_pins,
)
from gdsfactory.gpdk import LAYER


def test_add_ports() -> None:
    c = gf.Component()
    s = c << gf.components.straight()
    c.add_ports(s.ports)
    assert len(c.ports) == 2, len(c.ports)


def test_add_ports_from_pins() -> None:
    x = 1.235
    c = gf.components.straight(length=x)
    c = c.copy()
    c.flatten()
    c = gf.add_pins.add_pins_container(c)

    gdspath = c.write_gds(with_metadata=False)
    add_ports = partial(
        add_ports_from_markers_inside, pin_layer=LAYER.PORT, inside=True
    )

    c2 = gf.import_gds(gdspath, post_process=(add_ports,))
    assert c2.ports["o1"].center[0] == 0, c2.ports["o1"].center[0]
    assert c2.ports["o2"].center[0] == x, c2.ports["o2"].center[0]


def test_add_ports_from_pins_path() -> None:
    x = 1.239
    c = gf.components.straight(length=x)
    c = gf.add_pins.add_pins_siepic_container(c)
    assert c.ports["o1"].center[0] == 0
    assert c.ports["o2"].center[0] == x, c.ports["o2"].center[0]
    gdspath = c.write_gds(with_metadata=False)
    c2 = gf.import_gds(gdspath, post_process=(add_ports_from_siepic_pins,))  # type: ignore[arg-type]
    assert c2.ports["o1"].center[0] == 0, c2.ports["o1"].center[0]
    assert math.isclose(c2.ports["o2"].center[0], x), c2.ports["o2"].center[0]


def test_add_ports_from_labels() -> None:
    x = 1.238
    c = gf.components.straight(length=x)
    c = gf.add_pins.add_pins_container(c)
    port_width = c.ports["o1"].width
    gdspath = c.write_gds(with_metadata=False)
    add_ports = partial(
        add_ports_from_labels, port_layer=LAYER.TEXT, port_width=port_width
    )

    c2 = gf.import_gds(gdspath, post_process=(add_ports,))
    assert c2.ports["o1"].center[0] == 0
    assert c2.ports["o2"].center[0] == x, c2.ports["o2"].center[0]


class TestAddPortWithKwargs:
    @pytest.fixture
    def component_and_ref(self) -> tuple[gf.Component, gf.ComponentReference]:
        c = gf.Component()
        straight_ref = c << gf.components.straight()
        return c, straight_ref

    def test_port_type(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        c.add_port(port=straight_ref.ports["o1"], name="o1", port_type="placement")
        assert c.ports["o1"].port_type == "placement", (
            "Port type should be changed to 'placement'"
        )

    def test_cross_section(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        xs = gf.cross_section.strip()
        c.add_port(port=straight_ref.ports["o1"], name="o1", cross_section=xs)
        assert "cross_section" in c.ports["o1"].info, (
            "Cross section should be stored in port info"
        )
        assert c.ports["o1"].info["cross_section"] == xs.name, (
            f"Cross section name should be {xs.name}"
        )

    def test_port_type_and_cross_section(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        xs = gf.cross_section.metal1()
        c.add_port(
            port=straight_ref.ports["o1"],
            name="o1",
            port_type="electrical",
            cross_section=xs,
        )
        assert c.ports["o1"].port_type == "electrical", (
            "Port type should be 'electrical'"
        )
        assert c.ports["o1"].info["cross_section"] == xs.name, (
            f"Cross section should be {xs.name}"
        )

    def test_electrical_to_optical(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, _ = component_and_ref
        c.add_port(
            name="elec1", center=(0, 0), width=1, layer=(1, 0), port_type="electrical"
        )
        c.add_port(port=c.ports["elec1"], name="opt1", port_type="optical")
        assert c.ports["opt1"].port_type == "optical", (
            "Port type should be changed from electrical to optical"
        )

    def test_preserve_defaults(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        original_port_type = straight_ref.ports["o1"].port_type
        c.add_port(port=straight_ref.ports["o1"], name="o1")
        assert c.ports["o1"].port_type == original_port_type, (
            f"Port type should be preserved as {original_port_type}"
        )

    def test_width(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        c.add_port(port=straight_ref.ports["o1"], name="o1", width=2.5)
        assert c.ports["o1"].width == 2.5, "Port width should be changed to 2.5"

    def test_center(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        new_center = (10.0, 20.0)
        c.add_port(port=straight_ref.ports["o1"], name="o1", center=new_center)
        assert c.ports["o1"].center == new_center, (
            f"Port center should be changed to {new_center}"
        )

    def test_orientation(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        c.add_port(port=straight_ref.ports["o1"], name="o1", orientation=90)
        assert c.ports["o1"].orientation == 90, (
            "Port orientation should be changed to 90"
        )

    def test_layer(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        c.add_port(port=straight_ref.ports["o1"], name="o1", layer=(2, 0))
        assert c.ports["o1"].layer.layer == 2, "Port layer should be changed to layer 2"

    def test_multiple_properties(
        self, component_and_ref: tuple[gf.Component, gf.ComponentReference]
    ) -> None:
        c, straight_ref = component_and_ref
        c.add_port(
            port=straight_ref.ports["o1"],
            name="new_port",
            width=3.0,
            orientation=45,
            port_type="electrical",
        )
        assert c.ports["new_port"].width == 3.0, "Width should be 3.0"
        assert c.ports["new_port"].orientation == 45, "Orientation should be 45"
        assert c.ports["new_port"].port_type == "electrical", (
            "Port type should be electrical"
        )
