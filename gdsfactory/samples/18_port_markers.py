"""You can define a function to add pins."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_triangle
from gdsfactory.component import Component


def test_pins_custom(**kwargs) -> Component:
    """You can define the `pins_function` that we use to add markers to each port."""
    return gf.components.straight(**kwargs, decorator=add_pins_triangle)


if __name__ == "__main__":
    c = test_pins_custom()
    c.show(show_ports=False)
