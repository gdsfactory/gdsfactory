"""

When we decorate a function with autoname we can also pass a flat `with_ports` that will add port markers to our component

"""
import pp
from pp.add_pins import add_pins_triangle


def test_pins():
    c = pp.c.waveguide(pins=True)
    return c


def test_pins_custom():
    """ We can even define the `pins_function` that we use to add markers to each port

    """
    c = pp.c.waveguide(pins=True, pins_function=add_pins_triangle)
    return c


if __name__ == "__main__":
    c = test_pins()
    c = test_pins_custom()
    pp.qp(c)
    pp.show(c)
