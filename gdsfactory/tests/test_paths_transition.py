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


def test_path_transitions():
    import gdsfactory as gf

    # Create our first CrossSection
    X1 = gf.CrossSection()
    X1.add(width=1.2, offset=0, layer=2, name="wg", ports=("o1", "o2"))
    X1.add(width=2.2, offset=0, layer=3, name="etch")
    X1.add(width=1.1, offset=3, layer=1, name="wg2")

    # Create the second CrossSection that we want to transition to
    X2 = gf.CrossSection()
    X2.add(width=1, offset=0, layer=2, name="wg", ports=("o1", "o2"))
    X2.add(width=3.5, offset=0, layer=3, name="etch")
    X2.add(width=3, offset=5, layer=1, name="wg2")

    # To show the cross-sections, let's create two Paths and
    # create Devices by extruding them
    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)
    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    # Place both cross-section Devices and quickplot them
    c = gf.Component()
    c << wg1
    wg2ref = c << wg2
    wg2ref.movex(7.5)

    # Create the transitional CrossSection
    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")
    # Create a Path for the transitional CrossSection to follow
    P3 = gf.path.straight(length=15, npoints=100)

    # Use the transitional CrossSection to create a Component
    straight_transition = gf.path.extrude(P3, Xtrans)

    P4 = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    wg_trans = gf.path.extrude(P4, Xtrans)

    c = gf.Component()
    wg1_ref = c << wg1  # First cross-section Component
    wg2_ref = c << wg2
    wgt_ref = c << wg_trans

    wgt_ref.connect("o1", wg1_ref.ports["o2"])
    wg2_ref.connect("o1", wgt_ref.ports["o2"])

    wg3 = c << straight_transition
    wg3.movey(10)
    return c


if __name__ == "__main__":
    c = test_path_transitions()
    c.show()

    # test_path_transition_class()
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
