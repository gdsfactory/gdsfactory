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


def test_import_twice():
    c0 = gf.Component("parent")
    c0 << gf.c.mzi_arms()
    gdspath1 = c0.write_gds("extra/mzi.gds")

    c1 = gf.import_gds(gdspath1)  # IMPORT
    c2 = gf.import_gds(gdspath1)  # IMPORT

    assert len(c1.references) == len(c2.references)
    assert len(c1.polygons) == len(c2.polygons)
    assert len(c1.labels) == len(c2.labels)
    assert len(c1.hash_geometry()) == len(c2.hash_geometry())


def test_import_thrice():
    c0 = gf.Component("parent")
    c0 << gf.c.mzi_arms()
    gdspath1 = c0.write_gds("extra/mzi.gds")

    c = gf.Component()
    c << gf.import_gds(gdspath1)  # IMPORT
    c << gf.import_gds(gdspath1)  # IMPORT
    c << gf.import_gds(gdspath1)  # IMPORT

    gdspath2 = c.write_gds("extra/mzis.gds")
    geometry.check_duplicated_cells(gdspath2)


if __name__ == "__main__":
    test_import_twice()
    # test_build_first()
    # test_import_first()

    # gf.clear_cache()
    # c0 << gf.c.mzi_arms()

    # gdspath1 = c0.write_gds("extra/mmi.gds")

    # c = gf.Component("parent")
    # c0 = gf.c.mmi1x2()
    # gdspath1 = "extra/mmi.gds"
    # c1 = gf.import_gds(gdspath1)  # IMPORT
    # c2 = gf.import_gds(gdspath1)  # IMPORT

    # c << c0
    # c << c1
    # c << c2

    # gdspath2 = c.write_gds("extra/mzis.gds")
    # geometry.check_duplicated_cells(gdspath2)
    # c.show()
