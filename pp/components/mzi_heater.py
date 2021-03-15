from pp.cell import cell
from pp.components.mzi import mzi
from pp.components.waveguide_heater import wg_heater_connected
from pp.types import ComponentFactory


@cell
def mzi_with_heater(
    waveguide_delta_length: ComponentFactory = wg_heater_connected, **kwargs
):
    return mzi(waveguide_delta_length=waveguide_delta_length, **kwargs)


if __name__ == "__main__":
    c = mzi_with_heater()
    c.show()
