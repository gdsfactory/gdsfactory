"""Lets add pins to each cell from the fab a PDK.

"""
import dataclasses

import pp
from pp.add_pins import add_outline, add_pins
from pp.tech import TECH, Layer, Library, Waveguide


@dataclasses.dataclass
class Metal1(Waveguide):
    width: float = 2.0
    width_wide: float = 10.0
    auto_widen: bool = False
    layer: Layer = (30, 0)
    radius: float = 10.0


METAL1 = Metal1()

TECH.waveguide.metal1 = METAL1


def post_init(component) -> None:
    """Fab specific functions over a component."""
    add_pins(component)
    add_outline(component)


LIBRARY = Library(name="fab_a", post_init=post_init)
LIBRARY.register([pp.c.mmi2x2, pp.c.mmi1x2, pp.c.mzi])
LIBRARY.register(sw=pp.c.straight)


if __name__ == "__main__":
    F = LIBRARY
    F.settings.mmi1x2.width_mmi = 5
    c = F.get_component("mzi")
    c.show()
