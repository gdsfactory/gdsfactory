"""Straight Doped PIN waveguide."""

import gdsfactory as gf
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import pin

straight_pin = gf.partial(straight, cross_section=pin)


if __name__ == "__main__":

    c = straight_pin()
    print(c.ports.keys())
    c.show()
