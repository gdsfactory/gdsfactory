from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.straight_heater_doped_rib()
    c = gf.routing.add_pads_top(c, port_names=("top_e1", "bot_e1"))
    c = gf.routing.add_fiber_array(c)

    # c = gf.components.mzi_phase_shifter_top_heater_metal()
    c.show()

    # scene = c.to_3d()
    # scene.show()
