import gdsfactory as gf
from gdsfactory.components.mzi import mzi
from gdsfactory.components.straight_heater import straight_with_heater
from gdsfactory.types import ComponentFactory


@gf.cell
def mzi_phase_shifter(
    phase_shifter: ComponentFactory = straight_with_heater,
    length_x: float = 200.0,
    **kwargs
):
    return mzi(straight_horizontal_top=phase_shifter, length_x=length_x, **kwargs)


if __name__ == "__main__":
    c = mzi_phase_shifter()
    c.show()
    print(c.name)
