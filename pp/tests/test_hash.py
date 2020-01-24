from pp.compare_cells import hash_cells
from pp.components.mzi2x2 import mzi2x2
import pp
import gdspy


def main():
    c = mzi2x2()
    h0 = c.hash_geometry()
    gdspath1 = "{}.gds".format(c.name)
    gdspath2 = "{}_2.gds".format(c.name)
    gdspath3 = "{}_3.gds".format(c.name)
    pp.write_gds(c, gdspath1)
    c1 = pp.import_gds(gdspath1, overwrite_cache=True)
    dh1 = hash_cells(c1, {}, dbg=True)
    h1 = dh1[c1.name]
    pp.write_gds(c1, gdspath2)
    print(h0)

    c2 = pp.import_gds(gdspath2, overwrite_cache=True)
    dh2 = hash_cells(c2, {}, dbg=True)
    h2 = dh2[c2.name]
    pp.write_gds(c2, gdspath3)
    c3 = pp.import_gds(gdspath3, overwrite_cache=True)
    dh3 = hash_cells(c3, {}, dbg=True)
    h3 = dh3[c3.name]

    print(dh2)
    print()
    print()
    print(dh3)
    print()
    print()

    print(h1)
    print(h2)
    print(h3)
    print()
    print(gdspy.gdsii_hash(gdspath1))
    print(gdspy.gdsii_hash(gdspath2))
    print(gdspy.gdsii_hash(gdspath3))


if __name__ == "__main__":
    main()
