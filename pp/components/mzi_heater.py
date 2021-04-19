from pp.cell import cell
from pp.components.mzi import mzi
from pp.components.straight_heater import wg_heater_connected
from pp.types import ComponentFactory


@cell
def mzi_with_heater(
    straight_delta_length: ComponentFactory = wg_heater_connected, **kwargs
):
    return mzi(straight_delta_length=straight_delta_length, **kwargs)


if __name__ == "__main__":
    c = mzi_with_heater()
    c.show()
