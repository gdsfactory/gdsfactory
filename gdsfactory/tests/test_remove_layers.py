from __future__ import annotations

import numpy as np

import gdsfactory as gf


@gf.cell
def rectangles3():
    c = gf.Component()
    c << gf.c.rectangle(size=(4, 4), layer=(1, 0))
    c << gf.c.rectangle(size=(4, 4), layer=(2, 0))
    c.distribute()
    return c


def test_remove_layers():
    c0 = rectangles3()
    assert np.isclose(c0.area(), 16.0 * 2)

    c1 = c0.remove_layers([(1, 0)])
    assert np.isclose(c1.area(), 16.0)


if __name__ == "__main__":
    # test_remove_layers()
    c0 = rectangles3()
    # assert np.isclose(c0.area(), 16.0 * 2)

    c1 = c0.remove_layers([(1, 0)])
    # c1 = c0.remove_layers([(1, 0)])
    # assert np.isclose(c1.area(), 16.0)
    # c0._cell.filter(
    #     [(1, 0)],
    # )
    # c0.show()
    c1.show()
