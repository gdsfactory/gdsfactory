from __future__ import annotations

import numpy as np

import gdsfactory as gf


@gf.cell
def rectangles0():
    c = gf.Component()
    r1 = gf.c.rectangle(size=(4, 4))
    r1.name = "r1"
    c << r1
    return c


@gf.cell
def rectangles1():
    c = gf.Component()
    r1 = gf.c.rectangle(size=(4, 2))
    r1.name = "r1"
    c << r1
    return c


def test_import_gds_cell_with_name_conflicts() -> None:
    """Imports specific cell."""
    pass


if __name__ == "__main__":
    c0 = rectangles0()
    gdspath0 = c0.write_gds()

    c1 = rectangles1()
    gdspath1 = c1.write_gds()

    c0 = gf.import_gds(gdspath0)
    assert np.isclose(c0.area(), 16.0)

    c1 = gf.import_gds(gdspath1)
    assert np.isclose(c1.area(), 8.0)

    c3 = gf.grid([c0, c1])
    assert np.isclose(c3.area(), 8.0 + 16)
    c3.show()
