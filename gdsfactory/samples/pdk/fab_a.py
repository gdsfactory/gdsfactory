"""Lets for example customize the default gf.PDK

Fab A is mostly defined using Metal layers in GDS layer (30, 0)

The metal layer traces are 2um wide

"""

import gdsfactory as gf
from gdsfactory.cross_section import strip
from gdsfactory.difftest import difftest

WIDTH = 2
LAYER = (30, 0)

fab_a_metal = gf.partial(strip, width=WIDTH, layer=LAYER)
fab_a_metal.__name__ = "fab_a_metal"


straight = gf.partial(gf.components.straight, cross_section=fab_a_metal)
bend_euler = gf.partial(gf.components.bend_euler, cross_section=fab_a_metal)
mmi1x2 = gf.partial(
    gf.components.mmi1x2,
    cross_section=fab_a_metal,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=3 * WIDTH,
)
mzi = gf.partial(gf.components.mzi, cross_section=fab_a_metal, splitter=mmi1x2)
gc = gf.partial(
    gf.components.grating_coupler_elliptical_te, layer=LAYER, wg_width=WIDTH
)


def test_waveguide():
    c = gf.components.straight(cross_section=fab_a_metal)
    difftest(c)


if __name__ == "__main__":

    # c = gf.components.straight(length=20, cross_section=fab_a_metal)
    c = mzi()
    wg_gc = gf.routing.add_fiber_array(
        component=c, grating_coupler=gc, cross_section=fab_a_metal
    )
    wg_gc.show()
