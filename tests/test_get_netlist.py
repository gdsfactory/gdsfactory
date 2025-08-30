from __future__ import annotations

import pytest

import gdsfactory as gf


def test_netlist_simple() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=2)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect("o1", c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 2


def test_netlist_simple_width_mismatch_throws_error() -> None:
    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=1)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect("o1", c1.ports["o2"], allow_width_mismatch=True)
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    with pytest.warns(UserWarning):
        c.get_netlist()


def test_netlist_complex() -> None:
    c = gf.components.ring_single()
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 6, len(netlist["instances"])


def test_get_netlist_cell_array() -> None:
    rows = 3
    component_to_array = gf.components.straight(length=10)
    c = gf.components.array(
        component_to_array,
        column_pitch=100,
        columns=1,
        rows=rows,
        add_ports=True,
    )
    n = c.get_netlist(allow_multiple=True)
    n_ports_expected = 2 * rows
    assert len(c.ports) == n_ports_expected, (
        f"Expected {n_ports_expected} ports on component. Got {len(c.ports)}"
    )
    assert len(n["instances"]) == 1, (
        f"Expected only one instance for array. Got {len(n['instances'])}"
    )
    inst_name = c.insts[0].name
    assert len(n["ports"]) == n_ports_expected, (
        f"Expected {n_ports_expected} ports in netlist. Got {len(n['ports'])}"
    )
    for ib in range(rows):
        for port in component_to_array.ports:
            expected_port_name = f"{port.name}_{ib + 1}_1"
            expected_lower_port_name = f"{inst_name}<0.{ib}>,{port.name}"
            assert expected_port_name in n["ports"]
            assert n["ports"][expected_port_name] == expected_lower_port_name

    inst = next(n["instances"].values())
    n_rows = inst["array"]["rows"]
    n_columns = inst["array"]["columns"]
    assert n_rows == rows and n_columns == 1, (
        f"Expected {n_rows=}={rows} and {n_columns=}=1"
    )


def test_get_netlist_cell_array_no_ports() -> None:
    rows = 3
    c = gf.components.array(
        gf.components.straight(length=10),
        columns=1,
        column_pitch=100,
        rows=rows,
        add_ports=False,
    )
    n = c.get_netlist(allow_multiple=True)
    assert len(c.ports) == 0, (
        f"Expected no ports on component with add_ports=False. Got {len(c.ports)}"
    )
    assert len(n["ports"]) == 0, (
        f"Expected no ports in netlist with add_ports=False. Got {len(n['ports'])}"
    )
    assert len(n["instances"]) == 1, (
        f"Expected only one instance for array. Got {len(n['instances'])}"
    )
    inst = next(n["instances"].values())
    assert inst["array"]["columns"] == 1 and inst["array"]["rows"] == rows


def test_get_netlist_cell_array_connecting() -> None:
    c = gf.components.array(
        gf.components.straight(length=100), columns=5, rows=1, column_pitch=100
    )
    with pytest.warns(UserWarning):
        # because the component-array has automatic external ports, we assume no internal self-connections
        # we expect a ValueError to be thrown where the serendipitous connections are
        c.get_netlist(allow_multiple=False)


def test_get_netlist_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    unconnected_optical_port_warnings = netlist["warnings"]["optical"][
        "unconnected_ports"
    ]
    assert len(unconnected_optical_port_warnings) == 1
    assert len(unconnected_optical_port_warnings[0]["ports"]) == 4


def test_get_netlist_promoted() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)
    c.add_port("t1", port=i1.ports["o2"])
    c.add_port("t2", port=i2.ports["o1"])
    c.add_port("t3", port=i3.ports["o1"])
    c.add_port("t4", port=i3.ports["o2"])
    netlist = c.get_netlist()
    links = netlist["nets"]
    ports = netlist["ports"]
    expected_ports = {"t1": "i1,o2", "t2": "i2,o1", "t3": "i3,o1", "t4": "i3,o2"}

    assert len(links) == 1
    assert ports == expected_ports
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    assert "warnings" not in netlist


def test_get_netlist_close_enough_fails() -> None:
    """Move connection 1nm outwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovex(1)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 0


def test_get_netlist_close_enough_orthogonal_fails() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovey(0.001)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 0


def test_get_netlist_close_enough_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.rotate(angle=0.01)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_throws_error_bad_rotation() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.rotate(90)
    with pytest.warns(UserWarning):
        c.get_netlist()


def test_get_netlist_tiny() -> None:
    c = gf.Component()
    cc = gf.components.straight(length=0.002)
    i1 = c.add_ref(cc, "i1")
    i2 = c.add_ref(cc, "i2")
    i3 = c.add_ref(cc, "i3")
    i4 = c.add_ref(cc, "i4")

    i2.connect("o2", i1.ports["o1"])
    i3.connect("o2", i2.ports["o1"])
    i4.connect("o2", i3.ports["o1"])

    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 3
    # cpairs = list(connections.items())
    # extracted_port_pair = set(cpairs[0])
    # expected_port_pair = {'i2,o2', 'i1,o1'}
    # assert extracted_port_pair == expected_port_pair


def test_get_netlist_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i3 = c.add_ref(gf.components.wire_straight(), "i3")
    i2.connect("e2", i1.ports["e1"])
    i3.dmovey(-100)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_rotated_joint() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.rotate(45)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_different_widths() -> None:
    """Move connection 1nm inwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(width=1, cross_section="metal1"), "i1")
    i2 = c.add_ref(gf.components.straight(width=10, cross_section="metal1"), "i2")
    i2.connect("e2", i1.ports["e1"], allow_width_mismatch=True)
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1, len(links)
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_transformed() -> None:
    rotation_value = 35
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(rotation_value)
    i2.connect("o2", i1.ports["o1"])

    # perform the initial sanity checks on the netlist
    netlist = c.get_netlist()
    links = netlist["nets"]
    assert len(links) == 1, len(links)
    extracted_port_pair = set(links[0].values())
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_virtual_insts() -> None:
    """Test that virtual instances are included in the netlist."""
    c = gf.Component()
    cross_section = "strip"
    i1 = c.create_vinst(gf.components.straight(length=10, cross_section=cross_section))
    i2 = c.create_vinst(gf.components.straight(length=9, cross_section=cross_section))
    bend = c.create_vinst(
        gf.components.bend_euler_all_angle(angle=90, cross_section=cross_section)
    )
    bend.connect("o1", i1.ports["o2"])
    i2.connect("o1", bend.ports["o2"])
    c.add_port("o1", port=i1.ports["o1"])
    c.add_port("o2", port=i2.ports["o2"])
    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 3, (
        f"Expected 3 instances in netlist. Got {len(netlist['instances'])}"
    )
    assert len(netlist["nets"]) == 2, (
        f"Expected 2 nets in netlist. Got {len(netlist['nets'])}"
    )
    assert len(netlist["ports"]) == 2, (
        f"Expected 2 ports in netlist. Got {len(netlist['ports'])}"
    )


def test_get_netlist_virtual_cell() -> None:
    """Test that get_netlist works with virtual cells."""
    c = gf.ComponentAllAngle()
    cross_section = "strip"
    i1 = c.create_vinst(gf.components.straight(length=10, cross_section=cross_section))
    i2 = c.create_vinst(gf.components.straight(length=9, cross_section=cross_section))
    bend = c.create_vinst(
        gf.components.bend_euler_all_angle(angle=33, cross_section=cross_section)
    )
    bend.connect("o1", i1.ports["o2"])
    i2.connect("o1", bend.ports["o2"])
    c.add_port("o1", port=i1.ports["o1"])
    c.add_port("o2", port=i2.ports["o2"])

    netlist = c.get_netlist()
    assert len(netlist["instances"]) == 3, (
        f"Expected 3 instances in netlist. Got {len(netlist['instances'])}"
    )
    assert len(netlist["nets"]) == 2, (
        f"Expected 2 nets in netlist. Got {len(netlist['nets'])}"
    )
    assert len(netlist["ports"]) == 2, (
        f"Expected 2 ports in netlist. Got {len(netlist['ports'])}"
    )
