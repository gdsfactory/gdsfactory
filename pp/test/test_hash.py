import gdspy

import pp
from pp.compare_cells import hash_cells
from pp.components.mzi2x2 import mzi2x2


def debug():
    c = mzi2x2()
    h0 = c.hash_geometry()

    gdspath1 = "{}.gds".format(c.name)
    gdspath2 = "{}_2.gds".format(c.name)
    gdspath3 = "{}_3.gds".format(c.name)
    pp.write_gds(c, gdspath1)

    c1 = pp.import_gds(gdspath1, overwrite_cache=True)

    c2 = pp.import_gds(gdspath2, overwrite_cache=True)

    c3 = pp.import_gds(gdspath3, overwrite_cache=True)

    dbg = False
    dh1 = hash_cells(c1, {}, dbg=dbg)
    dh2 = hash_cells(c2, {}, dbg=dbg)
    dh3 = hash_cells(c3, {}, dbg=dbg)

    h1 = dh1[c1.name]
    h2 = dh2[c2.name]
    h3 = dh3[c3.name]
    print(h1)
    print(h2)
    print(h3)

    print(h0)
    print(gdspy.gdsii_hash(gdspath1))
    print(gdspy.gdsii_hash(gdspath2))
    print(gdspy.gdsii_hash(gdspath3))


def test_hash() -> None:
    c1 = pp.c.waveguide(length=10)
    c2 = pp.c.waveguide(length=11)
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2


if __name__ == "__main__":
    debug()
