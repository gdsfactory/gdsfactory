import numpy as np

import gdsfactory as gf


def test_length_delay_snake() -> None:
    length = 200.0
    c = gf.c.delay_snake(n=2, length=length, length0=50, cross_section="strip")
    length_computed = c.area(layer=(1, 0)) / 0.5
    np.isclose(length, length_computed)


def test_length_delay_snake2() -> None:
    length = 200.0
    c = gf.c.delay_snake2(length=length, cross_section="strip")
    length_computed = c.area("WG") / 0.5
    np.isclose(length, length_computed)


def test_delay_snake_sbend_length() -> None:
    length = 200.0
    c = gf.c.delay_snake_sbend(length=length, cross_section="strip")
    length_computed = c.area(layer=(1, 0)) / 0.5
    np.isclose(length, length_computed)
