"""

When we decorate a function with autoname we can also pass a flat `with_ports` that will add port markers to our component

"""
import pp
from pp.add_pins import add_pins_triangle


def test_with_pins():
    c = pp.c.waveguide(with_pins=True)
    return c


def test_with_pins_custom():
    """ We can even define the `add_pins_function` that we use to add markers to each port

    """
    c = pp.c.waveguide(with_pins=True, add_pins_function=add_pins_triangle)
    return c


if __name__ == "__main__":
    c = test_with_pins()
    c = test_with_pins_custom()
    pp.qp(c)
    pp.show(c)
