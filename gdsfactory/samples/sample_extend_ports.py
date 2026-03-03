import gdsfactory as gf

if __name__ == "__main__":
    p = gf.path.straight()

    # Add a few "sections" to the cross-section.
    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
    s1 = gf.Section(width=2, offset=2, layer=(2, 0))
    s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
    x = gf.CrossSection(sections=(s0, s1, s2))

    c = gf.path.extrude(p, cross_section=x)
    c = gf.c.extend_ports(c, cross_section=x, auto_taper=False)

    c = gf.Component()
    c.add_port(
        name="o1",
        center=(0, 0),
        width=1,
        orientation=180,
        cross_section=x,
        register_cross_section_factory=True,
    )

    pdk = gf.get_active_pdk()
    pdk.cross_sections.keys()
    c.show()
