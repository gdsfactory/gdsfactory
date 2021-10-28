import gdsfactory as gf


def test_port_by_orientation() -> gf.Component:
    c = gf.c.nxn(west=1, north=2, east=3, south=4)
    p = c.port_by_orientation_cw(key="W0")
    assert p.name == "o1"
    return c


def test_port_by_orientation_ref() -> gf.Component:
    c = gf.Component()
    nxn = gf.c.nxn(west=1, north=2, east=3, south=4)
    ref = c << nxn
    ref.rotate(+90)
    c.add_ports(ref.ports)
    p1 = ref.port_by_orientation_cw(key="W0")
    p2 = ref.port_by_orientation_cw(key="W1")
    assert p1.name == "o2", p1.name
    assert p2.name == "o3", p2.name
    return c


if __name__ == "__main__":
    # c = test_port_by_orientation_ref()
    # c = gf.Component()
    # ref = c << nxn
    # ref.rotate(+90)
    # c.add_ports(ref.ports)
    # p = ref.port_by_orientation_cw(key="W0")
    # c.show()

    c = gf.c.nxn(west=1, north=2, east=3, south=4)
    p = c.port_by_orientation_cw(key="W0")
    c.pprint_ports()
    c.show()
