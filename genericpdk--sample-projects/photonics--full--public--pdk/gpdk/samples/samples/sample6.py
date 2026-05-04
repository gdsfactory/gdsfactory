"""Pack."""

import gdsfactory as gf


@gf.cell
def sample6_cross_section():
    p = gf.path.straight()

    # Add a few "sections" to the cross-section
    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
    s1 = gf.Section(width=2, offset=2, layer=(2, 0))
    s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
    x = gf.CrossSection(sections=(s0, s1, s2))

    return gf.path.extrude(p, cross_section=x)
