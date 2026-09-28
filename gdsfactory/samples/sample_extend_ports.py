import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    p = gf.path.straight()

    # Add a few "sections" to the cross-section.
    s0 = ((1, 0), -0.5, 0.5)
    s1 = ((2, 0), 1.0, 3.0)
    s2 = ((2, 0), -3.0, -1.0)
    x = gf.cross_section.cross_section(width=None, sections=(s0, s1, s2))

    c = gf.path.extrude(p, cross_section=x)
    c = gf.c.extend_ports(c, cross_section=x, auto_taper=False)

    c = gf.Component()
    c.add_port(
        name="o1",
        center=(0, 0),
        width=1,
        orientation=180,
        cross_section=x,
    )

    pdk = gf.get_active_pdk()
    pdk.cross_sections.keys()
    c.show()
