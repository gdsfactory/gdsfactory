"""Some waveguides make unecessary crossings."""

import gdsfactory as gf
from gdsfactory.samples.big_device import big_device


if __name__ == "__main__":
    c = big_device()
    c = gf.routing.add_fiber_single(component=c)
    c.show()
