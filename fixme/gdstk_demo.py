import gdstk


class Component(gdstk.Cell):
    pass


def rectangle():
    r1 = Component("FIRST")
    rect = gdstk.rectangle((0, 0), (2, 1))
    r1.add(rect)

    r2 = gdstk.Cell("r2")
    rect = gdstk.rectangle((0, 0), (2, 2))
    r2.add(rect)

    c = gdstk.Cell("Combined")
    ref1 = gdstk.Reference(r1)
    ref2 = gdstk.Reference(r2)
    c.add(ref1)
    c.add(ref2)

    c.flatten()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    lib = gdstk.Library()

    c = rectangle()
    lib.add(c)

    for ref in c.references:
        lib.add(ref.cell)
    lib.write_gds("a.gds")

    gf.show("a.gds")
