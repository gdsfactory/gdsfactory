import gdsfactory
from gdsfactory.difftest import difftest
from gdsfactory.gdsdiff.gdsdiff import gdsdiff


def test_gdsdiff() -> None:
    c1 = gdsfactory.components.straight(length=5.0)
    c2 = gdsfactory.components.straight(length=6.0)
    c = gdsdiff(c1, c2)
    difftest(c)
