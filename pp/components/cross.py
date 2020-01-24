import pp


@pp.autoname
def cross(length=10, width=3, layer=0):
    """Generates a right-angle cross from two rectangles of specified length and width.

    Args:
        length: float Length of the cross from one end to the other
        width: float Width of the arms of the cross
        layer: int, array-like[2], or set Specific layer(s) to put polygon geometry on

    .. plot::
      :include-source:

      import pp

      c = pp.c.cross(length=10, width=3, layer=0)
      pp.plotgds(c)

    """

    c = pp.Component()
    R = pp.c.rectangle(size=(width, length), layer=layer)
    r1 = c.add_ref(R).rotate(90)
    r2 = c.add_ref(R)
    r1.center = (0, 0)
    r2.center = (0, 0)
    return c


if __name__ == "__main__":
    c = cross()
    pp.show(c)
