import gdsfactory as gf


def test_rotate() -> None:
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid


if __name__ == "__main__":
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid
    c2r.show()
