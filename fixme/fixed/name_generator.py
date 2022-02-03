import gdsfactory as gf


@gf.cell
def rectangles(widths: gf.types.Floats) -> gf.Component:
    c = gf.Component()
    for width in widths:
        c << gf.components.rectangle(size=(width, width))

    c.distribute()
    return c


if __name__ == "__main__":
    c1 = rectangles((i for i in range(5)))
    c2 = rectangles((i for i in range(6)))

    print(c1.name)
    print(c2.name)
    c1.show()
