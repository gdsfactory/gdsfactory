import gdsfactory as gf
from gdsfactory.difftest import difftest
from gdsfactory.geometry.offset import offset


def test_offset() -> None:
    c = gf.components.ring()
    co = offset(c, distance=0.5)
    difftest(co)
