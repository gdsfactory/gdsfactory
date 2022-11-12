import gdsfactory as gf
from gdsfactory.add_pins import add_pins, add_pins_siepic
from gdsfactory.add_ports import (
    add_ports_from_markers_inside,
    add_ports_from_siepic_pins,
)


def test_add_ports_dict() -> None:
    c = gf.Component()
    s = c << gf.components.straight()
    c.add_ports(s.ports)
    assert len(c.ports) == 2, len(c.ports)


def test_add_ports_list() -> None:
    c = gf.Component()
    s = c << gf.components.straight()
    c.add_ports(s.get_ports_list())
    assert len(c.ports) == 2, len(c.ports)


def test_add_ports_from_pins(data_regression):
    c = gf.components.straight(decorator=add_pins)
    gdspath = c.write_gds()
    c2 = gf.import_gds(gdspath, decorator=add_ports_from_markers_inside)
    data_regression.check(c2.to_dict(with_ports=True))


def test_add_ports_from_pins_siepic(data_regression):
    c = gf.components.straight(decorator=add_pins_siepic)
    gdspath = c.write_gds()
    c2 = gf.import_gds(gdspath, decorator=add_ports_from_siepic_pins)
    data_regression.check(c2.to_dict(with_ports=True))


if __name__ == "__main__":
    # test_add_ports_list()
    # test_add_ports_dict()

    c = gf.components.straight(decorator=add_pins)
    gdspath = c.write_gds()
    c2 = gf.import_gds(gdspath, decorator=add_ports_from_markers_inside)

    assert len(c2.ports) == 2

    x1, y1 = c.ports["o1"].center
    x2, y2 = c2.ports["o1"].center
    assert x1 == x2, f"{x1} {x2}"
    assert y1 == y2, f"{y1} {y2}"
