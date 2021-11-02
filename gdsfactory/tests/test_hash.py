"""FIXME. Tests are failing for python3.7
"""


import gdsfactory as gf
from gdsfactory.component import hash_file


def test_hash_geometry() -> None:
    """Test geometric hash of the GDS points."""
    c1 = gf.components.straight(length=10)
    c2 = gf.components.straight(length=11)
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2


def _test_hash_array_file():
    """Test hash of a component with an array of references."""
    gf.clear_cache()
    c = gf.Component("array")
    wg = gf.components.straight(length=3.2)
    c.add_array(wg)
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    assert h == "71d476075cf081b4099c1eea1c8984a1", h


def _test_hash_file():
    """Test hash of the saved GDS file."""
    gf.clear_cache()
    c = gf.components.straight()
    c.add_label("hi")
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    assert h == "f2228aed8141f447e601ce93a6219415", h


if __name__ == "__main__":
    # test_hash_geometry()
    _test_hash_file()
    _test_hash_array_file()
