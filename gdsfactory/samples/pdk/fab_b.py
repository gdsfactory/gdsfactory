"""Lets for example customize the default gf.PDK

Fab B is mostly uses optical layers but the waveguides required many cladding layers to avoid tiling, dopants...

Lets say that the waveguides are defined in layer (2, 0) and are 0.3um wide

"""

import gdsfactory as gf
from gdsfactory.cross_section import strip
from gdsfactory.difftest import difftest

WIDTH = 0.3
LAYER = (2, 0)
LAYERS_CLADDING = ((71, 0), (68, 0))


fab_b_metal = gf.partial(
    strip,
    width=WIDTH,
    layer=LAYER,
    layers_cladding=LAYERS_CLADDING,
)
fab_b_metal.__name__ = "fab_b_metal"


straight = gf.partial(gf.components.straight, cross_section=fab_b_metal)
bend_euler = gf.partial(gf.components.bend_euler, cross_section=fab_b_metal)
mmi1x2 = gf.partial(
    gf.components.mmi1x2,
    cross_section=fab_b_metal,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=3 * WIDTH,
)
mzi = gf.partial(gf.components.mzi, cross_section=fab_b_metal, splitter=mmi1x2)
gc = gf.partial(
    gf.components.grating_coupler_elliptical_te, layer=LAYER, wg_width=WIDTH
)


def test_waveguide():
    c = gf.components.straight(cross_section=fab_b_metal)
    difftest(c)


if __name__ == "__main__":

    # c = gf.components.straight(length=20, cross_section=fab_b_metal)
    c = mzi()
    wg_gc = gf.routing.add_fiber_array(
        component=c, grating_coupler=gc, cross_section=fab_b_metal
    )
    wg_gc.show()
