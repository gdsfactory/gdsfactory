"""You can also use the fiber array routing functions for connecting to pads."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.samples.big_device import big_device

if __name__ == "__main__":
    w = h = 18 * 50
    c = big_device(nports=10)
    c = gf.routing.add_fiber_array(
        component=c,
        cross_section="metal3",
        grating_coupler="pad",
        gc_port_name="e1",
        get_input_labels_function=None,
        with_loopback=False,
        straight_separation=15,
        # bend='wire_corner'
    )
    c.show()
