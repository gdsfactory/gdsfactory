import gdsfactory as gf


def test_path_transition_class():
    P = gf.path.straight(length=10, npoints=101)

    X1 = gf.CrossSection()
    X1.add(width=1, offset=0, layer=gf.LAYER.WG, name="core", ports=("o1", "o2"))
    X1.add(width=3, offset=0, layer=gf.LAYER.SLAB90)

    X2 = gf.CrossSection()
    X2.add(width=3, offset=0, layer=gf.LAYER.WG, name="core", ports=("o1", "o2"))

    T = gf.path.transition(X1, X2)
    c3 = gf.path.extrude(P, T)

    sections = {
        section["name"]: section for section in X1.sections if "name" in section
    }

    assert c3.ports["o1"].name == sections["core"]["ports"][0]
    assert c3.ports["o1"].layer == sections["core"]["layer"]
    assert c3.ports["o1"].orientation == 180
    assert c3.ports["o1"].port_type == sections["core"]["port_types"][0]


def test_path_transition_function():
    P = gf.path.straight(length=10, npoints=101)
    X1 = gf.cross_section.cross_section(width=1)
    X2 = gf.cross_section.cross_section(width=3)
    T = gf.path.transition(X1, X2)
    P = gf.path.straight(length=10, npoints=101)
    c3 = gf.path.extrude(P, T)

    sections = {
        section["name"]: section for section in X1.sections if "name" in section
    }

    assert c3.ports["o1"].name == sections["_default"]["ports"][0]
    assert c3.ports["o1"].layer == sections["_default"]["layer"]
    assert c3.ports["o1"].orientation == 180
    assert c3.ports["o1"].port_type == sections["_default"]["port_types"][0]


if __name__ == "__main__":
    test_path_transition_class()
    # test_path_transition_function()

    # X1 = gf.cross_section.cross_section(width=1)
    # X2 = gf.cross_section.cross_section(width=3)
    # T = gf.path.transition(X1, X2)
    # P = gf.path.straight(length=10, npoints=101)
    # c3 = gf.path.extrude(P, T)

    # sections = {
    #     section["name"]: section for section in X1.sections if "name" in section
    # }

    # assert c3.ports["o1"].name == sections["_default"]["ports"][0]
    # assert c3.ports["o1"].layer == sections["_default"]["layer"]
    # assert c3.ports["o1"].orientation == 180
    # assert c3.ports["o1"].port_type == sections["_default"]["port_types"][0]
    # assert c3.ports["o1"].port_type == X1.port_type
