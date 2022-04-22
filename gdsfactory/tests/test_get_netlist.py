import gdsfactory as gf


def test_get_netlist_cell_array() -> None:
    c = gf.c.array()
    n = c.get_netlist_dict()
    assert len(n.keys()) == 5


if __name__ == "__main__":
    c = gf.c.array()
    n = c.get_netlist_dict()
    print(len(n.keys()))
