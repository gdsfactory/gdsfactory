from pp.cell import cell
from pp.components.mzi import mzi
from pp.components.straight_heater import straight_with_heater
from pp.types import ComponentFactory


@cell
def mzi_with_heater(
    straight_delta_length: ComponentFactory = straight_with_heater, **kwargs
):
    return mzi(straight_delta_length=straight_delta_length, **kwargs)


if __name__ == "__main__":
    c = mzi_with_heater()
    c.show()
