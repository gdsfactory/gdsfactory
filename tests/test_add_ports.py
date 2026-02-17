from __future__ import annotations

import math
from functools import partial

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


def test_add_port_with_kwargs() -> None:
    """Test that add_port with port parameter can modify port_type and cross_section."""
    # Create a component with a reference
    c = gf.Component()
    straight_ref = c << gf.components.straight()
    
    # Test 1: Change port_type
    c.add_port(port=straight_ref.ports["o1"], name="o1", port_type="placement")
    assert c.ports["o1"].port_type == "placement", f"Expected 'placement', got {c.ports['o1'].port_type}"
    
    # Test 2: Change cross_section (this should set the info)
    c2 = gf.Component()
    straight_ref2 = c2 << gf.components.straight()
    xs = gf.cross_section.strip()
    c2.add_port(port=straight_ref2.ports["o1"], name="o1", cross_section=xs)
    assert "cross_section" in c2.ports["o1"].info, "cross_section should be in port info"
    assert c2.ports["o1"].info["cross_section"] == xs.name, f"Expected '{xs.name}', got {c2.ports['o1'].info['cross_section']}"
    
    # Test 3: Change both port_type and cross_section
    c3 = gf.Component()
    straight_ref3 = c3 << gf.components.straight()
    xs3 = gf.cross_section.metal1()
    c3.add_port(
        port=straight_ref3.ports["o1"], 
        name="o1", 
        port_type="electrical",
        cross_section=xs3
    )
    assert c3.ports["o1"].port_type == "electrical", f"Expected 'electrical', got {c3.ports['o1'].port_type}"
    assert c3.ports["o1"].info["cross_section"] == xs3.name, f"Expected '{xs3.name}', got {c3.ports['o1'].info['cross_section']}"
    
    # Test 4: Default port_type="optical" should not change the port_type
    c4 = gf.Component()
    straight_ref4 = c4 << gf.components.straight()
    original_port_type = straight_ref4.ports["o1"].port_type
    c4.add_port(port=straight_ref4.ports["o1"], name="o1")
    assert c4.ports["o1"].port_type == original_port_type, f"Expected '{original_port_type}', got {c4.ports['o1'].port_type}"
