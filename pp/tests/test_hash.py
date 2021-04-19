import pp


def test_hash() -> None:
    # FIXME, figure out straight hash issue
    # c1 = pp.components.straight(length=10)
    # c2 = pp.components.straight(length=11)
    c1 = pp.components.rectangle(size=(4, 0))
    c2 = pp.components.rectangle(size=(3, 0))
    h1 = c1.hash_geometry()
    h2 = c2.hash_geometry()
    assert h1 != h2


if __name__ == "__main__":
    test_hash()
