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


def test_add_port_with_kwargs(subtests) -> None:
    """Test that add_port with port parameter can modify all kwargs."""
    test_cases = [
        "port_type",
        "cross_section",
        "port_type_and_cross_section",
        "electrical_to_optical",
        "preserve_defaults",
        "width",
        "center",
        "orientation",
        "layer",
        "multiple_properties",
    ]

    for test_case in test_cases:
        with subtests.test(msg=test_case):
            c = gf.Component()
            straight_ref = c << gf.components.straight()

            match test_case:
                case "port_type":
                    # Test changing port_type
                    c.add_port(port=straight_ref.ports["o1"], name="o1", port_type="placement")
                    assert c.ports["o1"].port_type == "placement", "Port type should be changed to 'placement'"

                case "cross_section":
                    # Test changing cross_section
                    xs = gf.cross_section.strip()
                    c.add_port(port=straight_ref.ports["o1"], name="o1", cross_section=xs)
                    assert "cross_section" in c.ports["o1"].info, "Cross section should be stored in port info"
                    assert c.ports["o1"].info["cross_section"] == xs.name, f"Cross section name should be {xs.name}"

                case "port_type_and_cross_section":
                    # Test changing both port_type and cross_section
                    xs = gf.cross_section.metal1()
                    c.add_port(
                        port=straight_ref.ports["o1"],
                        name="o1",
                        port_type="electrical",
                        cross_section=xs
                    )
                    assert c.ports["o1"].port_type == "electrical", "Port type should be 'electrical'"
                    assert c.ports["o1"].info["cross_section"] == xs.name, f"Cross section should be {xs.name}"

                case "electrical_to_optical":
                    # Test changing port from electrical to optical explicitly
                    c.add_port(name="elec1", center=(0, 0), width=1, layer=(1, 0), port_type="electrical")
                    c.add_port(port=c.ports["elec1"], name="opt1", port_type="optical")
                    assert c.ports["opt1"].port_type == "optical", "Port type should be changed from electrical to optical"

                case "preserve_defaults":
                    # Test default behavior - port_type not explicitly changed when not provided
                    original_port_type = straight_ref.ports["o1"].port_type
                    c.add_port(port=straight_ref.ports["o1"], name="o1")
                    assert c.ports["o1"].port_type == original_port_type, f"Port type should be preserved as {original_port_type}"

                case "width":
                    # Test changing width when copying a port
                    c.add_port(port=straight_ref.ports["o1"], name="o1", width=2.5)
                    assert c.ports["o1"].width == 2.5, "Port width should be changed to 2.5"

                case "center":
                    # Test changing center when copying a port
                    new_center = (10.0, 20.0)
                    c.add_port(port=straight_ref.ports["o1"], name="o1", center=new_center)
                    assert c.ports["o1"].center == new_center, f"Port center should be changed to {new_center}"

                case "orientation":
                    # Test changing orientation when copying a port
                    c.add_port(port=straight_ref.ports["o1"], name="o1", orientation=90)
                    assert c.ports["o1"].orientation == 90, "Port orientation should be changed to 90"

                case "layer":
                    # Test changing layer when copying a port
                    c.add_port(port=straight_ref.ports["o1"], name="o1", layer=(2, 0))
                    assert c.ports["o1"].layer.layer == 2, "Port layer should be changed to layer 2"

                case "multiple_properties":
                    # Test changing multiple properties together
                    c.add_port(
                        port=straight_ref.ports["o1"],
                        name="new_port",
                        width=3.0,
                        orientation=45,
                        port_type="electrical"
                    )
                    assert c.ports["new_port"].width == 3.0, "Width should be 3.0"
                    assert c.ports["new_port"].orientation == 45, "Orientation should be 45"
                    assert c.ports["new_port"].port_type == "electrical", "Port type should be electrical"
