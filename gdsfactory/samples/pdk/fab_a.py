"""Lets for example customize the default gdsfactory PDK

Fab A is mostly defined using wide layers in GDS layer (1, 0)

The metal layer traces are 2um wide

"""

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_siepic
from gdsfactory.cross_section import strip

WIDTH = 2
LAYER = (1, 0)

fab_a_metal = gf.partial(
    strip, width=WIDTH, layer=LAYER, layer_bbox=(68, 0), decorator=add_pins_siepic
)

straight = gf.partial(gf.components.straight, cross_section=fab_a_metal)
bend_euler = gf.partial(gf.components.bend_euler, cross_section=fab_a_metal)
mmi1x2 = gf.partial(
    gf.components.mmi1x2,
    cross_section=fab_a_metal,
    width=WIDTH,
    width_taper=WIDTH,
    width_mmi=3 * WIDTH,
    decorator=add_pins_siepic,
)
ring_single = gf.partial(gf.components.ring_single, cross_section=fab_a_metal)
mzi = gf.partial(gf.components.mzi, cross_section=fab_a_metal, splitter=mmi1x2)
gc = gf.partial(
    gf.components.grating_coupler_elliptical_te,
    layer=LAYER,
    wg_width=WIDTH,
    decorator=add_pins_siepic,
)


if __name__ == "__main__":

    # c = gf.components.straight(length=20, cross_section=fab_a_metal)
    # c = mzi()
    c = ring_single()
    wg_gc = gf.routing.add_fiber_array(
        component=c, grating_coupler=gc, cross_section=fab_a_metal
    )
    wg_gc.show(show_ports=False)
