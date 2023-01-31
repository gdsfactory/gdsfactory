from __future__ import annotations

import numpy as np

import gdsfactory as gf


def test_import_gds_cell() -> None:
    """Imports specific cell."""
    c0 = gf.c.rectangle()
    gdspath = c0.write_gds()

    gf.clear_cache()

    # top cell with rectangle reference
    c1 = gf.import_gds(gdspath)
    assert np.isclose(c1.area(), 8.0)
    assert c1.name == "rectangle"

    # compass is top cell
    c2 = gf.import_gds(gdspath, cellname="compass")
    assert np.isclose(c2.area(), 8.0)
    assert c2.name == "compass"


if __name__ == "__main__":
    c0 = gf.c.rectangle()
    gdspath = c0.write_gds()

    gf.clear_cache()

    # top cell with rectangle reference
    c1 = gf.import_gds(gdspath)
    assert np.isclose(c1.area(), 8.0)
    assert c1.name == "rectangle"

    # compass is top cell
    c2 = gf.import_gds(gdspath, cellname="compass")
    assert np.isclose(c2.area(), 8.0)
    assert c2.name == "compass"
