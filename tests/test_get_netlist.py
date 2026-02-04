from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.get_netlist import PortCenterMatcher, SmartPortMatcher, legacy_namer


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
    n = c.get_netlist(on_multi_connect="ignore", instance_namer=legacy_namer)
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

    inst = next(iter(n["instances"].values()))
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
    n = c.get_netlist(on_multi_connect="ignore")
    assert len(c.ports) == 0, (
        f"Expected no ports on component with add_ports=False. Got {len(c.ports)}"
    )
    assert len(n["ports"]) == 0, (
        f"Expected no ports in netlist with add_ports=False. Got {len(n['ports'])}"
    )
    assert len(n["instances"]) == 1, (
        f"Expected only one instance for array. Got {len(n['instances'])}"
    )
    inst = next(iter(n["instances"].values()))
    assert inst["array"]["columns"] == 1 and inst["array"]["rows"] == rows


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
    netlist = c.get_netlist(port_matcher=SmartPortMatcher(position_tolerance_dbu=0.0))
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
    netlist = c.get_netlist(port_matcher=PortCenterMatcher())
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
    i1 = c.add_ref_off_grid(
        gf.components.straight(length=10, cross_section=cross_section)
    )
    i2 = c.add_ref_off_grid(
        gf.components.straight(length=9, cross_section=cross_section)
    )
    bend = c.add_ref_off_grid(
        gf.components.bend_euler_all_angle(angle=90, cross_section=cross_section),
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
    i1 = c.add_ref_off_grid(
        gf.components.straight(length=10, cross_section=cross_section)
    )
    i2 = c.add_ref_off_grid(
        gf.components.straight(length=9, cross_section=cross_section)
    )
    bend = c.add_ref_off_grid(
        gf.components.bend_euler_all_angle(angle=33, cross_section=cross_section),
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


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_get_netlist_array_roundtrip(rotation: int, mirror: bool) -> None:
    """Test that get_netlist -> from_yaml round-trip is idempotent for arrays.

    The array pitch values in the netlist should be in the local (pre-transform)
    frame, so that from_yaml can reconstruct the same component and produce
    the same netlist on re-serialization.
    """
    c1 = gf.Component()
    ref = c1.add_ref(gf.get_component("straight"), rows=8, columns=1, row_pitch=100)
    if mirror:
        ref.dmirror()
    if rotation:
        ref.rotate(rotation)
    n1 = c1.get_netlist()

    c2 = gf.read.from_yaml(n1)
    n2 = c2.get_netlist()

    inst1 = next(iter(n1["instances"].values()))
    inst2 = next(iter(n2["instances"].values()))
    assert inst1["array"] == inst2["array"], (
        f"Array config changed after round-trip with {rotation=}, {mirror=}.\n"
        f"  Before: {inst1['array']}\n"
        f"  After:  {inst2['array']}"
    )

    pl1 = next(iter(n1["placements"].values()))
    pl2 = next(iter(n2["placements"].values()))
    assert pl1 == pl2, (
        f"Placement changed after round-trip with {rotation=}, {mirror=}.\n"
        f"  Before: {pl1}\n"
        f"  After:  {pl2}"
    )


# Tests for _extract_nets_from_connects


def test_extract_nets_from_connects_empty() -> None:
    """Test with empty connects list."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    result = _extract_nets_from_connects([])
    assert result == {}


def test_extract_nets_from_connects_single() -> None:
    """Test with a single connection."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    connects = [{"p1": "inst1,o1", "p2": "inst2,o1"}]
    result = _extract_nets_from_connects(connects)
    assert len(result) == 1
    net = next(iter(result.values()))
    assert set(net) == {"inst1,o1", "inst2,o1"}


def test_extract_nets_from_connects_transitive() -> None:
    """Test transitive connectivity - the main bug case."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    # Test actual transitive case: A-B and B-C should form one net
    connects = [
        {"p1": "A,o1", "p2": "B,o1"},
        {"p1": "B,o1", "p2": "C,o1"},
    ]
    result = _extract_nets_from_connects(connects)
    assert len(result) == 1
    net = next(iter(result.values()))
    assert set(net) == {"A,o1", "B,o1", "C,o1"}


def test_extract_nets_from_connects_star_topology() -> None:
    """Test star topology where one port connects to multiple others."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    connects = [
        {"p1": "hub,o1", "p2": "spoke1,o1"},
        {"p1": "hub,o1", "p2": "spoke2,o1"},
        {"p1": "hub,o1", "p2": "spoke3,o1"},
    ]
    result = _extract_nets_from_connects(connects)
    assert len(result) == 1
    net = next(iter(result.values()))
    assert set(net) == {"hub,o1", "spoke1,o1", "spoke2,o1", "spoke3,o1"}


def test_extract_nets_from_connects_disjoint() -> None:
    """Test multiple disjoint nets."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    connects = [
        {"p1": "A,o1", "p2": "B,o1"},
        {"p1": "C,o1", "p2": "D,o1"},
    ]
    result = _extract_nets_from_connects(connects)
    assert len(result) == 2
    nets_as_sets = [set(net) for net in result.values()]
    assert {"A,o1", "B,o1"} in nets_as_sets
    assert {"C,o1", "D,o1"} in nets_as_sets


def test_extract_nets_from_connects_deterministic() -> None:
    """Test that net names are deterministic regardless of input order."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    connects1 = [
        {"p1": "A,o1", "p2": "B,o1"},
        {"p1": "B,o1", "p2": "C,o1"},
    ]
    connects2 = [
        {"p1": "C,o1", "p2": "B,o1"},
        {"p1": "B,o1", "p2": "A,o1"},
    ]
    result1 = _extract_nets_from_connects(connects1)
    result2 = _extract_nets_from_connects(connects2)

    # Same net name should be generated
    assert list(result1.keys()) == list(result2.keys())
    # Same ports should be in the net
    assert next(iter(result1.values())) == next(iter(result2.values()))


def test_extract_nets_from_connects_with_settings() -> None:
    """Test that extra keys in connects dict are ignored."""
    from gdsfactory.get_netlist import _extract_nets_from_connects

    connects = [
        {"p1": "A,o1", "p2": "B,o1", "settings": {"width1": 0.5, "width2": 0.6}},
    ]
    result = _extract_nets_from_connects(connects)
    assert len(result) == 1
    net = next(iter(result.values()))
    assert set(net) == {"A,o1", "B,o1"}
