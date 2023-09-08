from __future__ import annotations

import pandas as pd

import gdsfactory as gf
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


if __name__ == "__main__":
    c = gf.grid(
        [mzi_te()] * 2,
        add_ports_suffix=True,
        add_ports_prefix=False,
        decorator=gf.add_labels.add_labels_to_ports,
    )
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(gdspath, debug=debug)

    df = pd.read_csv(csvpath)
    c.show()
