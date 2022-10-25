import pytest

import gdsfactory as gf
from gdsfactory.decorators import flatten_invalid_refs
from gdsfactory.get_netlist import get_netlist_recursive


def test_get_netlist_cell_array() -> gf.Component:
    c = gf.components.array(
        gf.components.straight(length=10), spacing=(0, 100), columns=1, rows=5
    )
    n = c.get_netlist()
    assert len(c.ports) == 10
    assert not n["connections"]
    assert len(n["ports"]) == 10
    assert len(n["instances"]) == 5
    return c


def test_get_netlist_cell_array_connecting() -> gf.Component:
    c = gf.components.array(
        gf.components.straight(length=100), spacing=(100, 0), columns=5, rows=1
    )
    with pytest.raises(ValueError):
        # because the component-array has automatic external ports, we assume no internal self-connections
        # we expect a ValueError to be thrown where the serendipitous connections are
        c.get_netlist()

    return c


@gf.cell
def test_get_netlist_simple() -> gf.Component:
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
    return c


@gf.cell
def test_get_netlist_promoted() -> gf.Component:
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
    return c


@gf.cell
def test_get_netlist_close_enough() -> gf.Component:
    """Move connection 1nm inwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.movex(0.001)
    netlist = c.get_netlist(tolerance=2)
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_close_enough_fails() -> gf.Component:
    """Move connection 1nm outwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.movex(-0.001)
    netlist = c.get_netlist(tolerance=1)
    connections = netlist["connections"]
    assert len(connections) == 0
    return c


@gf.cell
def test_get_netlist_close_enough_orthogonal() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.movey(0.001)
    netlist = c.get_netlist(tolerance=2)
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_close_enough_orthogonal_fails() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.movey(0.001)
    netlist = c.get_netlist(tolerance=1)
    connections = netlist["connections"]
    assert len(connections) == 0
    return c


@gf.cell
def test_get_netlist_close_enough_both() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.move((0.001, 0.001))
    netlist = c.get_netlist(tolerance=2)
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_close_enough_rotated() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.rotate(angle=0.01, center="o2")
    netlist = c.get_netlist(tolerance=2)
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_throws_error_bad_rotation() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.rotate(angle=90, center="o2")
    with pytest.raises(ValueError):
        c.get_netlist(tolerance=2)

    return c


@gf.cell
def test_get_netlist_tiny() -> gf.Component:
    c = gf.Component()
    cc = gf.components.straight(length=0.002)
    i1 = c.add_ref(cc, "i1")
    i2 = c.add_ref(cc, "i2")
    i3 = c.add_ref(cc, "i3")
    i4 = c.add_ref(cc, "i4")

    i2.connect("o2", i1.ports["o1"])
    i3.connect("o2", i2.ports["o1"])
    i4.connect("o2", i3.ports["o1"])

    netlist = c.get_netlist(tolerance=5)
    connections = netlist["connections"]
    assert len(connections) == 3
    # cpairs = list(connections.items())
    # extracted_port_pair = set(cpairs[0])
    # expected_port_pair = {'i2,o2', 'i1,o1'}
    # assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_rotated() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,o2", "i1,o1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_electrical_simple() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i3 = c.add_ref(gf.components.wire_straight(), "i3")
    i2.connect("e2", i1.ports["e1"])
    i3.movey(-100)
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_electrical_rotated_joint() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.rotate(45, "e2")
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_electrical_allowable_offset() -> gf.Component:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.move((0.001, 0.001))
    netlist = c.get_netlist()
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair
    return c


@gf.cell
def test_get_netlist_electrical_different_widths() -> gf.Component:
    """Move connection 1nm inwards."""
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(width=1, cross_section="metal1"), "i1")
    i2 = c.add_ref(gf.components.straight(width=10, cross_section="metal1"), "i2")
    i2.move("e2", destination=i1.ports["e1"])
    i2.movex(0.001)
    netlist = c.get_netlist(tolerance=2)
    connections = netlist["connections"]
    assert len(connections) == 1
    cpairs = list(connections.items())
    extracted_port_pair = set(cpairs[0])
    expected_port_pair = {"i2,e2", "i1,e1"}
    assert extracted_port_pair == expected_port_pair
    return c


def test_get_netlist_transformed():
    rotation_value = 35
    cname = "test_get_netlist_transformed"
    c = gf.Component(cname)
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.rotate(rotation_value)
    i2.connect("o2", i1.ports["o1"])

    # flatten the oddly rotated refs
    c = flatten_invalid_refs(c)

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
    assert len(recursive_netlist) == 3
    # confirm that the child netlists have reference attributes properly set

    i1_cell_name = top_netlist["instances"]["i1"]["component"]
    i1_netlist = recursive_netlist[i1_cell_name]
    # currently for transformed netlists, the instance name of the inner cell is None
    assert i1_netlist["placements"][None]["rotation"] == rotation_value

    i2_cell_name = top_netlist["instances"]["i2"]["component"]
    i2_netlist = recursive_netlist[i2_cell_name]
    # currently for transformed netlists, the instance name of the inner cell is None
    assert i2_netlist["placements"][None]["rotation"] == rotation_value


if __name__ == "__main__":
    # c = gf.c.array()
    # n = c.get_netlist()
    # print(len(n.keys()))
    # c = test_get_netlist_cell_array()
    # c = test_get_netlist_cell_array_connecting()
    # c = test_get_netlist_simple()
    # c = test_get_netlist_promoted()
    # c = test_get_netlist_close_enough()
    # c = test_get_netlist_close_enough_orthogonal()
    # c = test_get_netlist_close_enough_fails()
    # c = test_get_netlist_close_enough_orthogonal_fails()
    # c = test_get_netlist_close_enough_both()
    # c = test_get_netlist_close_enough_rotated()
    # c = test_get_netlist_throws_error_bad_rotation()
    # c = test_get_netlist_tiny()
    # c = test_get_netlist_metal()
    c = test_get_netlist_electrical_different_widths()
    c.show()
