"""Lets add pins to each cell from the fab a PDK.

"""
import dataclasses

import gdsfactory
from gdsfactory.add_pins import add_outline, add_pins
from gdsfactory.tech import TECH, Layer, Library, Waveguide


@dataclasses.dataclass
class Metal1(Waveguide):
    width: float = 2.0
    width_wide: float = 10.0
    auto_widen: bool = False
    layer: Layer = (30, 0)
    radius: float = 10.0


METAL1 = Metal1()

TECH.waveguide.metal1 = METAL1


def decorator(component) -> None:
    """Fab specific functions over a component."""
    add_pins(component)
    add_outline(component)


mmi2x2 = gdsfactory.partial(gdsfactory.components.mmi2x2, decorator=decorator)
mmi1x2 = gdsfactory.partial(gdsfactory.components.mmi1x2, decorator=decorator)
mzi = gdsfactory.partial(gdsfactory.components.mzi, splitter=mmi1x2)

LIBRARY = Library(name="fab_a")
LIBRARY.register([mmi2x2, mmi1x2, mzi])


if __name__ == "__main__":
    F = LIBRARY
    F.settings.mmi1x2.width_mmi = 5
    c = F.get_component("mzi")
    c.show()
