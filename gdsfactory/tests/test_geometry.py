import gdsfactory as gf

r1 = (8, 8)
r2 = (11, 4)

angle_resolution = 2.5

c1 = gf.components.ellipse(radii=r1, layer=(1, 0), angle_resolution=angle_resolution)
c2 = gf.components.ellipse(radii=r2, layer=(1, 0), angle_resolution=angle_resolution)


def test_boolean_not() -> None:
    c4 = gf.geometry.boolean(c1, c2, operation="not", layer=(1, 0))
    assert int(c4.area()) == 87


def test_boolean_not_klayout() -> None:
    c3 = gf.geometry.boolean_klayout(c1, c2, operation="not", layer3=(1, 0))
    assert int(c3.area()) == 87


def test_boolean_or() -> None:
    c4 = gf.geometry.boolean(c1, c2, operation="or", layer=(1, 0))
    assert int(c4.area()) == 225


def test_boolean_or_klayout() -> None:
    c3 = gf.geometry.boolean_klayout(c1, c2, operation="or", layer3=(1, 0))
    assert int(c3.area()) == 225


def test_boolean_xor() -> None:
    c4 = gf.geometry.boolean(c1, c2, operation="xor", layer=(1, 0))
    assert int(c4.area()) == 111


def test_boolean_xor_klayout() -> None:
    c3 = gf.geometry.boolean_klayout(c1, c2, operation="xor", layer3=(1, 0))
    assert int(c3.area()) == 111


def test_boolean_and() -> None:
    c4 = gf.geometry.boolean(c1, c2, operation="and", layer=(1, 0))
    assert int(c4.area()) == 113


def test_boolean_and_klayout() -> None:
    c3 = gf.geometry.boolean_klayout(c1, c2, operation="and", layer3=(1, 0))
    assert int(c3.area()) == 113


if __name__ == "__main__":
    c3 = gf.geometry.boolean_klayout(c1, c2, operation="and", layer3=(1, 0))
    # c4 = gf.geometry.boolean(c1, c2, operation="and", layer=(1, 0))
    # print(int(c3.area()))
    # c4.show()
