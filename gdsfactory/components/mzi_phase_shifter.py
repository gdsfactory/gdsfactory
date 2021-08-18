import gdsfactory as gf
from gdsfactory.components.mzi import mzi
from gdsfactory.components.straight_heater import straight_heater_metal

mzi_phase_shifter = gf.partial(
    mzi, straight_horizontal_top=straight_heater_metal, length_x=320.0
)
mzi_phase_shifter.__name__ = "mzi_phase_shifter"


if __name__ == "__main__":
    c = mzi_phase_shifter()
    c.show()
    print(c.name)
