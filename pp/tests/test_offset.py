import pp
from pp.difftest import difftest
from pp.offset import offset


def test_offset() -> None:
    c = pp.components.ring()
    co = offset(c, distance=0.5)
    difftest(co)
