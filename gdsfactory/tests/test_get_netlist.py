import pytest

import gdsfactory as gf


def test_get_netlist_cell_array() -> None:
    c = gf.components.array(
        gf.components.straight(length=10), spacing=(0, 100), columns=1, rows=5
    )
    n = c.get_netlist()
    assert len(c.ports) == 10
    assert not n["connections"]
    assert len(n["ports"]) == 10
    assert len(n["instances"]) == 5


def test_get_netlist_cell_array_connecting() -> None:
    c = gf.components.array(
        gf.components.straight(length=100), spacing=(100, 0), columns=5, rows=1
    )
    with pytest.raises(ValueError):
        # because the component-array has automatic external ports, we assume no internal self-connections
        # we expect a ValueError to be thrown where the serendipitous connections are
        c.get_netlist()


def test_get_netlist_simple():
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


def test_get_netlist_promoted():
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


def test_get_netlist_close_enough():
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


def test_get_netlist_close_enough_orthogonal():
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


def test_get_netlist_close_enough_both():
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


def test_get_netlist_close_enough_rotated():
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


def test_get_netlist_throws_error_bad_rotation():
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.move("o2", destination=i1.ports["o1"])
    i2.rotate(angle=90, center="o2")
    with pytest.raises(ValueError):
        c.get_netlist(tolerance=2)


def test_get_netlist_tiny():
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


def test_get_netlist_rotated():
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


def test_get_netlist_electrical_simple():
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


def test_get_netlist_electrical_rotated_joint():
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


def test_get_netlist_electrical_allowable_offset():
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


if __name__ == "__main__":
    c = gf.c.array()
    n = c.get_netlist()
    print(len(n.keys()))
