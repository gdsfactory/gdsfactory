import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    s0 = ((2, 0), -0.6, 0.6)
    s1 = ((3, 0), -1.1, 1.1)
    s2 = ((1, 0), 2.45, 3.55)
    X1 = gf.cross_section.cross_section(width=None, sections=(s0, s1, s2))

    # Create the second Cross-section that we want to transition to.
    s0 = ((2, 0), -0.5, 0.5)
    s1 = ((3, 0), -1.75, 1.75)
    s2 = ((1, 0), 3.5, 6.5)
    X2 = gf.cross_section.cross_section(width=None, sections=(s0, s1, s2))

    # To show the cross-sections, let us now create two paths and create components by extruding them.
    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)
    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    # Place both cross-section components and quickplot them,
    # Quickplot is designed to create a wide variety of complex graphs with a simple, concise syntax, making it ideal for quick data exploration.
    c = gf.Component()
    wg1ref = c << wg1
    wg2ref = c << wg2
    wg2ref.movex(7.5)

    # Create the transitional cross-section.
    # Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")
    Xtrans = gf.path.transition(
        cross_section1=wg1["o2"].info["cross_section"],
        cross_section2=wg2["o1"].info["cross_section"],
        width_type="sine",
    )

    # Create a Path for the transitional cross-section to follow.
    P3 = gf.path.straight(length=15, npoints=100)

    # Use the transitional cross-section to create a component.
    c = gf.path.extrude_transition(P3, Xtrans)

    c.show()
