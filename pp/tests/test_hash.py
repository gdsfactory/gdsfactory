import pp
from pp.component import hash_file


def test_hash_geometry() -> None:
    # FIXME, figure out straight hash issue
    # c1 = pp.components.straight(length=10)
    # c2 = pp.components.straight(length=11)
    c1 = pp.components.rectangle(size=(4, 0))
    c2 = pp.components.rectangle(size=(3, 0))
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2


def test_hash_file():
    c = pp.c.straight()
    gdspath = c.write_gds()
    h = hash_file(gdspath)
    assert h == "c6d3387b8ea0de9838a2393dba1f56e6"


if __name__ == "__main__":
    # test_hash_geometry()
    test_hash_file()
