from __future__ import annotations

import numpy as np

import gdsfactory as gf


@gf.cell
def rectangles3() -> gf.Component:
    c = gf.Component()
    ref1 = c << gf.c.rectangle(size=(4, 4), layer=(1, 0))
    ref2 = c << gf.c.rectangle(size=(4, 4), layer=(2, 0))

    ref2.d.xmin = ref1.d.xmax + 10
    return c


def test_remove_layers() -> None:
    c0 = rectangles3()
    assert np.isclose(c0.area(layer=(1, 0)), 16.0), c0.area(layer=(1, 0))

    c1 = c0.remove_layers([(1, 0)])
    assert np.isclose(c1.area(layer=(2, 0)), 16.0), c1.area(layer=(2, 0))
