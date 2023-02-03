"""Straight Doped PIN waveguide."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.cross_section import rib

straight_rib = gf.partial(straight, cross_section=rib)


straight_rib_tapered = gf.partial(
    extend_ports,
    component=straight_rib,
    extension=taper_strip_to_ridge,
    port1="o2",
    port2="o1",
)


if __name__ == "__main__":
    # c = straight_rib()
    c = straight_rib_tapered()
    # c.plot_holoviews()
    c.show(show_ports=True)
