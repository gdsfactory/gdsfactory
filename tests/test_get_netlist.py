from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.get_netlist import get_netlist_recursive


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
    with pytest.raises(ValueError):
        c.get_netlist()


def test_netlist_complex() -> None:
    c = gf.components.mzi_arms()
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 4, len(netlist["instances"])


def test_get_netlist_cell_array() -> None:
    c = gf.components.array(
        gf.components.straight(length=10), spacing=(0, 100), columns=1, rows=5
    )
    n = c.get_netlist(allow_multiple=True)
    assert len(c.ports) == 10, len(c.ports)
    # assert not n["connections"], n["connections"]
    assert len(n["ports"]) == 2, len(n["ports"])
    assert len(n["instances"]) == 1, len(n["instances"])


def test_get_netlist_cell_array_connecting() -> None:
    c = gf.components.array(
        gf.components.straight(length=100), spacing=(100, 0), columns=5, rows=1
    )
    with pytest.raises(ValueError):
        # because the component-array has automatic external ports, we assume no internal self-connections
        # we expect a ValueError to be thrown where the serendipitous connections are
        c.get_netlist()


def test_get_netlist_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.movey(-100)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
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
    i3.movey(-100)
    c.add_port("t1", port=i1.ports["o2"])
    c.add_port("t2", port=i2.ports["o1"])
    c.add_port("t3", port=i3.ports["o1"])
    c.add_port("t4", port=i3.ports["o2"])
    netlist = c.get_netlist()
    connections = netlist["connections"]
    ports = netlist["ports"]
    expected_ports = {"t1": "i1,o2", "t2": "i2,o1", "t3": "i3,o1", "t4": "i3,o2"}

    assert len(connections) == 1
    assert ports == expected_ports
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    assert "warnings" not in netlist


def test_get_netlist_close_enough_fails() -> None:
    """Move connection 1nm outwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.movex(1)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 0


def test_get_netlist_close_enough_orthogonal_fails() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovey(0.001)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 0


def test_get_netlist_close_enough_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.drotate(angle=0.01)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_throws_error_bad_rotation() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.drotate(90)
    with pytest.raises(ValueError):
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
    connections = netlist["connections"]
    assert len(connections) == 3
    # cpairs = list(connections.items())
    # extracted_port_pair = set(cpairs[0])
    # expected_port_pair = {'i2,o2', 'i1,o1'}
    # assert extracted_port_pair == expected_port_pair


def test_get_netlist_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.drotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
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
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_rotated_joint() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.drotate(45)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


def test_get_netlist_electrical_different_widths() -> None:
    """Move connection 1nm inwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(width=1, cross_section="metal1"), "i1")
    i2 = c.add_ref(gf.components.straight(width=10, cross_section="metal1"), "i2")
    i2.connect("e2", i1.ports["e1"], allow_width_mismatch=True)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1, len(connections)
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair


@pytest.mark.skip("TODO")
def test_get_netlist_transformed() -> None:
    rotation_value = 35
    cname = "test_get_netlist_transformed"
    c = gf.Component(cname)
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.drotate(rotation_value)
    i2.connect("o2", i1.ports["o1"])

    # perform the initial sanity checks on the netlist
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair

    recursive_netlist = get_netlist_recursive(c)
    top_netlist = recursive_netlist[cname]
    # the recursive netlist should have 3 entries, for the top level and two rotated straights
    assert len(recursive_netlist) == 1, len(recursive_netlist)
    # confirm that the child netlists have reference attributes properly set

    i1_cell_name = top_netlist["instances"]["i1"]["component"]

    i1_netlist = recursive_netlist[i1_cell_name]
    assert i1_netlist["placements"][None]["rotation"] == rotation_value

    i2_cell_name = top_netlist["instances"]["i2"]["component"]
    i2_netlist = recursive_netlist[i2_cell_name]
    assert i2_netlist["placements"][None]["rotation"] == rotation_value


if __name__ == "__main__":
    test_get_netlist_electrical_rotated_joint()
    # test_get_netlist_electrical_different_widths()
    # test_netlist_simple_width_mismatch_throws_error()
    # test_get_netlist_cell_array()
    # test_get_netlist_transformed()
