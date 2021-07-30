import pp
from pp.components.mzi import mzi
from pp.components.straight_heater import straight_with_heater
from pp.types import ComponentFactory


@pp.cell
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
