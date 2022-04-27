"""Lets add pins to each cell from the fab a PDK.

"""

import gdsfactory as gf
from gdsfactory.add_pins import add_outline, add_pins
from gdsfactory.cross_section import strip
from gdsfactory.difftest import difftest

WIDTH = 2
LAYER = (30, 0)

fab_a_metal = gf.partial(strip, width=WIDTH, layer=LAYER)
fab_a_metal.__name__ = "fab_a_metal"


def test_waveguide() -> None:
    c = gf.components.straight(cross_section=fab_a_metal)
    difftest(c)


def decorator(component) -> None:
    """Fab specific functions over a component."""
    add_pins(component)
    add_outline(component)


mmi2x2 = gf.partial(gf.components.mmi2x2, decorator=decorator)
mmi1x2 = gf.partial(gf.components.mmi1x2, decorator=decorator)
bend_euler = gf.partial(gf.components.bend_euler, decorator=decorator)
straight = gf.partial(gf.components.straight, decorator=decorator)
mzi = gf.partial(gf.components.mzi, splitter=mmi1x2, bend=bend_euler, straight=straight)

cells = dict(mmi2x2=mmi2x2, mmi1x2=mmi1x2, mzi=mzi)


if __name__ == "__main__":
    c = mzi()
    c.show()
