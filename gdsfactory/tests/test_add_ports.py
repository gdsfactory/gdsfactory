import gdsfactory as gf


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


if __name__ == "__main__":
    test_add_ports_list()
    # test_add_ports_dict()
