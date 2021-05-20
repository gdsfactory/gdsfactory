import pp
from pp.components.mzi import mzi
from pp.components.straight_heater import straight_with_heater
from pp.types import ComponentFactory


@pp.cell_with_validator
def mzi_with_heater(
    straight_delta_length: ComponentFactory = straight_with_heater, **kwargs
):
    return mzi(straight_delta_length=straight_delta_length, **kwargs)


if __name__ == "__main__":
    c = mzi_with_heater()
    c.show()
