from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.samples.big_device import big_device
from gdsfactory.typings import Component

debug = False
nlabels = 12


def mzi_te(**kwargs) -> Component:
    """Returns MZI with TE grating couplers."""
    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_pads_top(c, port_names=["top_l_e4", "top_r_e4"])
    c = gf.routing.add_fiber_array(c, grating_coupler=gc, **kwargs)
    return c


def spirals() -> Component:
    cm = 10e3
    lengths = np.array([2, 4, 6]) * cm
    spirals = [gf.components.spiral_inner_io(length=length) for length in lengths]
    spirals_te = [gf.components.add_grating_couplers(c) for c in spirals]
    return gf.pack(spirals_te)[0]


@gf.cell
def demo_pack() -> Component:
    """Sample reticle."""
    c = gf.Component()
    _ = c << gf.c.rectangle(size=(22e3, 22e3), layer=(64, 0))
    components = [mzi_te()] + [spirals()] + [gf.routing.add_fiber_array(big_device())]
    ref = c << gf.pack(components)[0]
    c.add_ports(ref.ports)
    return c


if __name__ == "__main__":
    c = demo_pack()
    c.show()
