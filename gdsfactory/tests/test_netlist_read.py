import gdsfactory as gf
from gdsfactory.component import Component


def test_netlist_read() -> Component:
    filepath = gf.CONFIG["netlists"] / "mzi.yml"
    c = gf.read.from_yaml(filepath)

    # print(c.get_netlist().pretty())
    # print((c.get_netlist().connections.pretty()))
    # print(len(c.get_netlist().connections))
    # print(len(c.get_dependencies()))
    # assert len(c.get_netlist().connections) == 18

    assert len(c.get_dependencies()) == 5, len(c.get_dependencies())
    return c


def regenerate_regression_test() -> None:
    c = gf.components.mzi()

    filepath = gf.CONFIG["netlists"] / "mzi.yml"
    c.write_netlist(filepath)


if __name__ == "__main__":
    c = test_netlist_read()
    # test_netlist_read_full()
    # regenerate_regression_test()

    # c = test_netlist_read_full()
    c.show(show_ports=True)

    # n = c.get_netlist()
    # i = n["instances"]
    # b = i["bend_circular_R10_17.873_-5.5"]
    # layer = b["settings"]["layer"]
    # print(type(layer))
    # c.show(show_ports=True)
