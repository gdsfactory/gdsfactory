import gdsfactory
from gdsfactory.difftest import difftest
from gdsfactory.offset import offset


def test_offset() -> None:
    c = gdsfactory.components.ring()
    co = offset(c, distance=0.5)
    difftest(co)
