from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.mzi import mzi
from gdsfactory.components.straight_heater_metal import straight_heater_metal

mzi_phase_shifter = gf.partial(
    mzi, straight_x_top="straight_heater_metal", length_x=200
)

mzi_phase_shifter_top_heater_metal = gf.partial(
    mzi_phase_shifter, straight_x_top=straight_heater_metal
)

if __name__ == "__main__":
    # c = mzi_phase_shifter(splitter="mmi2x2")
    # c = mzi_phase_shifter_top_heater_metal(splitter="mmi2x2")
    # c = mzi_phase_shifter(splitter='mmi2x2')
    # c = mzi_phase_shifter(
    #     straight_x_top=gf.components.straight_pin, straight_x_bot=gf.components.straight_pin
    # )
    # c = mzi_phase_shifter(
    #     # straight_x_top=gf.components.straight_heater_doped_rib,
    #     straight_x_bot=gf.components.straight_heater_doped_rib,
    #     delta_length=20,
    #     length_x=600,
    # )
    # c = mzi_phase_shifter()

    # c = mzi_phase_shifter()
    c = mzi_phase_shifter()
    c.show(show_ports=True)
    print(c.name)
