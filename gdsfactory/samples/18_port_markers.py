"""You can define a function to add pins."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_triangle


if __name__ == "__main__":
    c = gf.components.straight(decorator=add_pins_triangle)
    c.show(show_ports=False)
