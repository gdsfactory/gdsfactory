import gdsfactory as gf


def test_ports_xsection() -> None:
    xs = gf.CrossSection(sections=(gf.Section(layer=(1, 0), width=2),))
    c = gf.get_component("bend_euler", angle=90, cross_section=xs)
    assert c.ports["e1"].cross_section


def test_ports_add() -> None:
    xs = gf.CrossSection(sections=(gf.Section(layer=(1, 1), width=2),))
    c = gf.Component()
    p1 = c.add_port(name="o1", center=(0, 0), width=1, orientation=0, cross_section=xs)
    p2 = c.add_port(
        name="o2",
        port=gf.Port(
            name="o2",
            center=(0, 0),
            width=1,
            orientation=0,
            cross_section=xs,
        ),
    )

    assert p1.layer == p2.layer
    assert p1.cross_section == p2.cross_section
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.width == p2.width
    assert p1.orientation == p2.orientation


def test_port_get_item():
    wg = gf.components.straight()

    assert wg["o1"] == wg[0]
    assert wg["o2"] == wg[1]

    wg_ref = wg.ref()
    assert wg_ref["o1"] == wg_ref[0]
    assert wg_ref["o2"] == wg_ref[1]


if __name__ == "__main__":
    test_ports_add()
