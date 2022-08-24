import gdsfactory as gf


def test_get_netlist_cell_array() -> None:
    c = gf.components.array()
    n = c.get_netlist()
    assert len(n.keys()) == 5


if __name__ == "__main__":
    c = gf.c.array()
    n = c.get_netlist()
    print(len(n.keys()))
