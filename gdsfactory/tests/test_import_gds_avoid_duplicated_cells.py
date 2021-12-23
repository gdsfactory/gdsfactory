"""avoid duplicated cell names when importing GDS files."""

import gdsfactory as gf
from gdsfactory import geometry


def test_import_first():
    c1 = gf.Component("parent")
    c1 << gf.c.mzi_arms()
    gdspath1 = c1.write_gds("extra/mzi.gds")

    gf.clear_cache()
    mzi1 = gf.import_gds(gdspath1)  # IMPORT
    c1 = gf.c.mzi_arms()  # BUILD

    c2 = gf.grid([mzi1, c1])
    gdspath2 = c2.write_gds("extra/mzi2.gds")
    geometry.check_duplicated_cells(gdspath2)


def test_build_first():
    c1 = gf.Component("parent")
    c1 << gf.c.mzi_arms()
    gdspath1 = c1.write_gds("extra/mzi.gds")

    gf.clear_cache()
    c1 = gf.c.mzi_arms()  # BUILD
    mzi1 = gf.import_gds(gdspath1)  # IMPORT

    c2 = gf.grid([mzi1, c1])
    gdspath2 = c2.write_gds("extra/mzi2.gds")
    geometry.check_duplicated_cells(gdspath2)


if __name__ == "__main__":

    # c1 = gf.Component("parent")
    # c1 << gf.c.mzi_arms()
    # gdspath1 = c1.write_gds("extra/mzi.gds")

    gdspath1 = "extra/mzi.gds"
    # gf.clear_cache()
    mzi1 = gf.import_gds(gdspath1)  # IMPORT
    c1 = gf.c.mzi_arms()  # BUILD

    c2 = gf.grid([mzi1, c1])
    gdspath2 = c2.write_gds("extra/mzi2.gds")
    c2.show()
    geometry.check_duplicated_cells(gdspath2)
