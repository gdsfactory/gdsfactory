import gdsfactory as gf


@gf.cell
def rectangles(widths: gf.types.Floats) -> gf.Component:
    c = gf.Component()
    for width in widths:
        c << gf.c.rectangle(size=(width, width))

    c.distribute()
    return c


def test_name_partial_functions():
    s1 = gf.partial(gf.c.straight)
    s2 = gf.partial(gf.c.straight, length=5)
    s3 = gf.partial(gf.c.straight, 5)

    m1 = gf.partial(gf.c.mzi, straight=s1)()
    m2 = gf.partial(gf.c.mzi, straight=s2)()
    m3 = gf.partial(gf.c.mzi, straight=s3)()

    # print(m1.name)
    # print(m2.name)
    # print(m3.name)

    assert (
        m2.name == m3.name
    ), f"{m2.name} different from {m2.name} while they are the same function"
    assert (
        m1.name != m2.name
    ), f"{m1.name} is the same {m2.name} while they are different functions"
    assert (
        m1.name != m3.name
    ), f"{m1.name} is the same {m3.name} while they are different functions"


def test_name_iterators():
    c1 = rectangles((i for i in range(5)))
    c2 = rectangles((i for i in range(6)))

    assert c1.name != c2.name


def test_name_int_float():
    """Ensure that passing the same int or float returns the same component"""
    c1 = gf.c.straight(length=5)
    c2 = gf.c.straight(length=5.0)

    assert c1.name == c2.name, f"{c1.name} does not match {c2.name}"


def test_float_point_errors():
    c1 = gf.c.straight(length=5.0 + 1e-20)  # any unit below pm disappears
    c2 = gf.c.straight(length=5.0)
    assert c1.name == c2.name, f"{c1.name} does not match {c2.name}"


# def test_name_different_signatures():
#     c1 = gf.c.compass()

#     @gf.cell
#     def compass(layer=(2, 0)):
#         return gf.c.compass(layer=layer)

#     c2 = compass()
#     assert c1.name != c2.name, f"{c1.name} should differ from {c2.name}"


if __name__ == "__main__":
    # test_name_partial_functions()
    # test_name_int_float()
    # test_name_different_signatures()

    c1 = gf.c.compass()

    @gf.cell
    def compass(layer=(2, 0)):
        return gf.c.compass(layer=layer)

    c2 = compass()
    assert c1.name != c2.name, f"{c1.name} should differ from {c2.name}"
