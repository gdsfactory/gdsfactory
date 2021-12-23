"""FIXME

import_gds
"""

import gdsfactory as gf


if __name__ == "__main__":
    c1 = gf.Component("parent")
    c1 << gf.c.mzi()
    gdspath1 = c1.write_gds("extra/mzi.gds")

    gf.clear_cache()
    c1 = gf.c.mzi()
    mzi1 = gf.import_gds(gdspath1)

    c2 = gf.grid([mzi1, c1])

    gdspath2 = c2.write_gds("extra/mzi2.gds")
    c2.show()

    from gdsfactory import geometry

    geometry.check_width(gdspath2)
