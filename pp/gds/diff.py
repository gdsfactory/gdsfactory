import pp
from pp.gdsdiff.gdsdiff import gdsdiff


if __name__ == "__main__":
    c1 = pp.c.mmi1x2(length_mmi=5)
    c2 = pp.c.mmi1x2(length_mmi=9)
    c3 = gdsdiff(c1, c2)
    pp.show(c3)
