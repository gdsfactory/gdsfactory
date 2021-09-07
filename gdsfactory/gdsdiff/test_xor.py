import gdsfactory as gf
from gdsfactory.gdsdiff.gdsdiff import gdsdiff


def test_differences():
    c1 = gf.c.straight(length=2)
    c2 = gf.c.straight(length=3)
    c = gdsdiff(c1, c2)
    assert c.references[-1].area() == 0.5


def test_no_differences():
    c1 = gf.c.straight(length=2)
    c2 = gf.c.straight(length=2)
    c = gdsdiff(c1, c2)
    assert c.references[-1].area() == 0


if __name__ == "__main__":
    # test_no_differences()
    # test_differences()
    c1 = gf.c.straight(length=2)
    c2 = gf.c.straight(length=2)
    c = gdsdiff(c1, c2)
    c.show()
