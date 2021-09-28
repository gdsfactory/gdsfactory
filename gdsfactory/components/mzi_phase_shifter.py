import gdsfactory as gf
from gdsfactory.components.mzi import mzi
from gdsfactory.components.straight_heater_metal import (
    straight_heater_metal,
    straight_heater_metal_90_90,
)

mzi_phase_shifter = gf.partial(
    mzi, straight_x_top=straight_heater_metal, length_x=320.0
)

mzi_phase_shifter_90_90 = gf.partial(
    mzi_phase_shifter, straight_x_top=straight_heater_metal_90_90
)

if __name__ == "__main__":
    c = mzi_phase_shifter_90_90()
    c.show()
    print(c.name)
