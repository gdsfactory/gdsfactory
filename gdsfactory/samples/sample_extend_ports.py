import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    p = gf.path.straight()

    # Add a few "sections" to the cross-section.
    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
    s1 = gf.Section(width=2, offset=2, layer=(2, 0))
    s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
    x = gf.cross_section.cross_section(
        width=s0.width,
        offset=s0.offset,
        layer=s0.layer,
        sections=(s1, s2),
        port_names=s0.port_names,
        port_types=s0.port_types,
    )

    c = gf.path.extrude(p, cross_section=x)
    c = gf.c.extend_ports(c, cross_section=x, auto_taper=False)

    c = gf.Component()
    c.add_port(
        name="o1",
        center=(0, 0),
        width=1,
        orientation=180,
        cross_section=x,
        register_cross_section=True,
    )

    pdk = gf.get_active_pdk()
    pdk.cross_sections.keys()
    c.show()
