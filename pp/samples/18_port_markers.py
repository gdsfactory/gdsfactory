"""

"""
import pp
from pp.add_pins import add_pins_triangle
from pp.component import Component


def test_pins_custom() -> Component:
    """You can define the `pins_function` that we use to add markers to each port"""
    c = pp.components.straight(length=11.1)
    add_pins_triangle(component=c)
    return c


if __name__ == "__main__":
    c = test_pins_custom()
    c.show(show_ports=False)
