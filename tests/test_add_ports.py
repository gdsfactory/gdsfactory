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
    assert c.ports["o1"].port_type == "placement"

    # Test 2: Change cross_section (this should set the info)
    c2 = gf.Component()
    straight_ref2 = c2 << gf.components.straight()
    xs = gf.cross_section.strip()
    c2.add_port(port=straight_ref2.ports["o1"], name="o1", cross_section=xs)
    assert "cross_section" in c2.ports["o1"].info
    assert c2.ports["o1"].info["cross_section"] == xs.name

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
    assert c3.ports["o1"].port_type == "electrical"
    assert c3.ports["o1"].info["cross_section"] == xs3.name

    # Test 4: Change port from electrical to optical explicitly
    c4 = gf.Component()
    # Create a port with electrical type
    c4.add_port(name="elec1", center=(0, 0), width=1, layer=(1, 0), port_type="electrical")
    # Now copy it but change to optical
    c4.add_port(port=c4.ports["elec1"], name="opt1", port_type="optical")
    assert c4.ports["opt1"].port_type == "optical"

    # Test 5: Default behavior - port_type not explicitly changed when not provided
    c5 = gf.Component()
    straight_ref5 = c5 << gf.components.straight()
    original_port_type = straight_ref5.ports["o1"].port_type
    c5.add_port(port=straight_ref5.ports["o1"], name="o1")
    assert c5.ports["o1"].port_type == original_port_type

    # Test 6: Change width when copying a port
    c6 = gf.Component()
    straight_ref6 = c6 << gf.components.straight(width=1.0)
    c6.add_port(port=straight_ref6.ports["o1"], name="o1", width=2.5)
    assert c6.ports["o1"].width == 2.5

    # Test 7: Change center when copying a port
    c7 = gf.Component()
    straight_ref7 = c7 << gf.components.straight()
    new_center = (10.0, 20.0)
    c7.add_port(port=straight_ref7.ports["o1"], name="o1", center=new_center)
    assert c7.ports["o1"].center == new_center

    # Test 8: Change orientation when copying a port
    c8 = gf.Component()
    straight_ref8 = c8 << gf.components.straight()
    c8.add_port(port=straight_ref8.ports["o1"], name="o1", orientation=90)
    assert c8.ports["o1"].orientation == 90

    # Test 9: Change layer when copying a port
    c9 = gf.Component()
    straight_ref9 = c9 << gf.components.straight()
    c9.add_port(port=straight_ref9.ports["o1"], name="o1", layer=(2, 0))
    # Layer comparison needs to account for layer enum
    assert c9.ports["o1"].layer.layer == 2

    # Test 10: Change multiple properties together
    c10 = gf.Component()
    straight_ref10 = c10 << gf.components.straight()
    c10.add_port(
        port=straight_ref10.ports["o1"],
        name="new_port",
        width=3.0,
        orientation=45,
        port_type="electrical"
    )
    assert c10.ports["new_port"].width == 3.0
    assert c10.ports["new_port"].orientation == 45
    assert c10.ports["new_port"].port_type == "electrical"
