import pp
from pp.difftest import difftest
from pp.gdsdiff.gdsdiff import gdsdiff


def test_gdsdiff() -> None:
    c1 = pp.c.waveguide(length=5.0)
    c2 = pp.c.waveguide(length=6.0)
    c = gdsdiff(c1, c2)
    difftest(c)
