import gdsfactory as gf


def test_route() -> None:
    c = gf.Component("sample_connect")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((100, 50))
    route = gf.routing.get_route(
        mmi1.ports["o3"],
        mmi2.ports["o1"],
        cross_section=gf.cross_section.metal1,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)


if __name__ == "__main__":
    c = gf.Component("sample_connect")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((100, 50))
    route = gf.routing.get_route(
        mmi1.ports["o3"],
        mmi2.ports["o1"],
        cross_section=gf.cross_section.metal1,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.show(show_ports=True)
