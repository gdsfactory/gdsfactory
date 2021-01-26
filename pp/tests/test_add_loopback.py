from pp.add_loopback import waveguide_with_loopback
from pp.component import Component
from pp.difftest import difftest


def test_add_loopback() -> Component:
    c = waveguide_with_loopback()
    difftest(c)
    return c
