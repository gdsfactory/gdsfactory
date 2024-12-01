from functools import partial

import gdsfactory as gf
from gdsfactory import cross_section as xs
from gdsfactory.generic_tech.layer_map import LAYER

###################
# Constants
###################
pad_size = (80, 80)
pad_pitch = 100

###################
# Cells
###################
straight_sc = partial(gf.c.straight, cross_section=xs.strip)
straight_heater_metal = partial(
    gf.c.straight_heater_metal,
    cross_section=xs.strip,
)
terminator = partial(gf.c.taper, width2=1)

grating_coupler_te = partial(gf.c.grating_coupler_te, cross_section="strip")
# grating_coupler_te = partial(gf.c.grating_coupler_te, cross_section=xs.strip)
pad = partial(gf.c.pad, size=pad_size, layer=LAYER.MTOP)

mmi1x2 = partial(
    gf.c.mmi1x2,
    width_taper=1,
    length_taper=10,
    length_mmi=10,
    width_mmi=2.5,
    cross_section=xs.strip,
)

#################################
# Cells that contain other cells
#################################

add_fiber_array_optical_south_electrical_north = partial(
    gf.c.add_fiber_array_optical_south_electrical_north,
    component=straight_heater_metal,
    pad=pad,
    grating_coupler=grating_coupler_te,
    cross_section_metal="metal_routing",
    pad_pitch=pad_pitch,
)
add_termination = partial(gf.c.add_termination, terminator=terminator)


if __name__ == "__main__":
    c = add_fiber_array_optical_south_electrical_north()
    c.show()
