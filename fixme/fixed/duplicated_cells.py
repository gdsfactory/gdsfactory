"""Trying to replicate a duplicated cell error coming from extruding a path.

"""

import gdsfactory as gf


@gf.cell
def sbend(radius: float = 5):
    """Dummy component"""
    c = gf.Component()
    path = gf.path.euler(radius=radius, angle=45, p=0.5, use_eff=False)
    path.append(gf.path.straight(length=10))
    path.append(gf.path.euler(radius=radius, angle=-45, p=0.5, use_eff=False))
    c << gf.path.extrude(path, gf.cross_section.strip)
    c.add_label("opt_te1550", layer=gf.LAYER.LABEL)
    return c


if __name__ == "__main__":
    c = gf.Component()
    s1 = c << sbend()
    s2 = c << sbend()
    s2.movex(1)

    c.show()

    gdspath = c.write_gds("mask.gds")
    csvpath = gf.mask.write_labels(gdspath)
