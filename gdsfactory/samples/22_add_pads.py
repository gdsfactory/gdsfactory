"""You can also use the fiber array routing functions for connecting to pads."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import PDK
from gdsfactory.samples.big_device_electrical import big_device

PDK.activate()

if __name__ == "__main__":
    w = h = 18 * 50
    c = big_device()
    c = gf.routing.add_pads_top(c, fanout_length=500)
    c.show()
