from gdsfactory.component import Component


def test_netlist_read() -> Component:
    import gdsfactory

    filepath = gdsfactory.CONFIG["netlists"] / "mzi.yml"
    c = gdsfactory.component_from_yaml(filepath)

    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    # assert len(c.get_netlist().connections) == 18
    print(len(c.get_dependencies()))

    assert len(c.get_dependencies()) == 4
    return c


def test_netlist_read_full() -> Component:
    import gdsfactory

    filepath = gdsfactory.CONFIG["netlists"] / "mzi_full.yml"
    c = gdsfactory.component_from_yaml(filepath)

    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    assert len(c.get_dependencies()) == 4
    return c


def regenerate_regression_test():
    import gdsfactory

    c = gdsfactory.components.mzi()
    filepath = gdsfactory.CONFIG["netlists"] / "mzi_full.yml"
    c.write_netlist(filepath, full_settings=True)

    filepath = gdsfactory.CONFIG["netlists"] / "mzi.yml"
    c.write_netlist(filepath, full_settings=False)


if __name__ == "__main__":
    test_netlist_read()
    # test_netlist_read_full()
    # regenerate_regression_test()

    # c = test_netlist_read_full()
    # c.show()

    # n = c.get_netlist()
    # i = n["instances"]
    # b = i["bend_circular_R10_17.873_-5.5"]
    # layer = b["settings"]["layer"]
    # print(type(layer))
    # c.show()
