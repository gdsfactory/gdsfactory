"""Sample route single."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    cell_names = [
        # 'die_frame_with_pads',
        # 'die_with_pads',
        # gf.c.grating_coupler_array(with_loopback=True),
        # 'greek_cross_with_pads',
        "loop_mirror",
        "mzi_pads_center",
        "straight_heater_meander",
        "straight_heater_meander_doped",
    ]
    c = gf.pack([gf.get_component(name) for name in cell_names])[0]
    c.show()
