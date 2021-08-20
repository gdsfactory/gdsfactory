"""Straight Doped PIN waveguide."""

import gdsfactory as gf
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.cross_section import pin

straight_pin = gf.partial(straight, cross_section=pin)

straight_pin_tapered = gf.partial(
    extend_ports,
    component=straight_pin,
    extension_factory=taper_strip_to_ridge,
    port1=2,
    port2=1,
)


if __name__ == "__main__":

    c = straight_pin_tapered()
    print(c.ports.keys())
    c.show()
