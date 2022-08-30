import gdsfactory as gf


def test_get_netlist_cell_array() -> None:
    c = gf.components.array()
    n = c.get_netlist()
    assert len(n.keys()) == 5


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


if __name__ == "__main__":
    c = gf.c.array()
    n = c.get_netlist()
    print(len(n.keys()))
