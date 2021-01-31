import pp


def test_hash() -> None:
    c1 = pp.c.waveguide(length=10)
    c2 = pp.c.waveguide(length=11)
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2
