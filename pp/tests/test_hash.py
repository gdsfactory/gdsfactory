import pp
from pp.component import hash_file


def test_hash_geometry() -> None:
    """Test geometric hash of the GDS points."""
    c1 = pp.components.straight(length=10)
    c2 = pp.components.straight(length=11)
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2


def test_hash_file():
    """Test hash of the saved GDS file."""
    c = pp.c.straight()
    c.add_label("hi")
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    print(h)
    assert h == "71655c3f7ab57e7a48b55683e8c1bfc4"


def test_hash_array_file():
    """FIXME, this needs some more gdspy fix."""
    c = pp.Component()
    wg = pp.c.straight()
    c.add_array(wg)
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    print(h)
    # assert h == "c6d3387b8ea0de9838a2393dba1f56e6"


if __name__ == "__main__":
    test_hash_geometry()
    # test_hash_file()
    # test_hash_array_file()
