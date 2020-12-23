def test_netlist_read(filepath=None):
    import pp

    filepath = filepath or pp.CONFIG["netlists"] / "mzi.yml"
    c = pp.component_from_yaml(filepath)
    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    # assert len(c.get_netlist().connections) == 18

    assert len(c.get_dependencies()) == 5
    return c


if __name__ == "__main__":
    import pp

    filepath = "ring_single.yml"
    c = pp.component_from_yaml(filepath)
    pp.show(c)
