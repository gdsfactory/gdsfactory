def test_netlist_read():
    import pp

    c = pp.component_from_yaml(pp.CONFIG["netlists"] / "mzi.yml")
    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    assert len(c.get_dependencies()) == 5
    # assert len(c.get_netlist().connections) == 18
    return c


if __name__ == "__main__":
    import pp

    c = test_netlist_read()
    pp.show(c)
