"""
Generate a spiral DOE
"""
import pp
from pp.routing.add_fiber_array import add_fiber_array
from pp.components.spiral_external_io import spiral_external_io


@pp.autoname
def SPIRAL(N=6, x=50.0):
    c = spiral_external_io(N=N, x_inner_length_cutback=x)
    return add_fiber_array(c, x_grating_offset=-200, fanout_length=30)


if __name__ == "__main__":
    c = SPIRAL()
    pp.show(c)
