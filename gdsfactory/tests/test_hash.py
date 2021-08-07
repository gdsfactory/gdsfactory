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
    print(h)
    assert h == "bec2ab8f157b429bd6ff210bedde6fe3"


def _test_hash_file():
    """Test hash of the saved GDS file."""
    gf.clear_cache()
    c = gf.components.straight()
    c.add_label("hi")
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    print(h)
    assert h == "71655c3f7ab57e7a48b55683e8c1bfc4"


if __name__ == "__main__":
    # test_hash_geometry()
    _test_hash_file()
    _test_hash_array_file()
