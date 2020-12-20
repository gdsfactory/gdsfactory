from typing import Callable, Tuple

import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component


@pp.cell
def disk(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle: int = 0,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: Callable = wg_strip,
    **kwargs
) -> Component:
    """ Disk Resonator

    Args:
       radius (float): Radius of the disk resonator
       gap (float): Distance between the bus waveguide and resonator
       wrap_angle (float): Angle in *radians* between 0 and pi (defaults to 0) that determines how much the bus waveguide wraps along the resonator.  0 corresponds to a straight bus waveguide, and pi corresponds to a bus waveguide wrapped around half of the resonator.
       parity (1 or -1): If 1, resonator to left of bus waveguide, if -1 resonator to the right
       port (tuple): Cartesian coordinate of the input port (x1, y1)
       direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate object

    Where in the above (x1,y1) is the same as the 'port' input, (x2, y2) is the end of the component, and 'dir1', 'dir2' are of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, *or* an angle in *radians*.
    'Direction' points *towards* the waveguide that will connect to it.

    Other Parameters:
       wg_width: 0.5
       wg_layer: pp.LAYER.WG[0]
       wg_datatype: pp.LAYER.WG[1]
       clad_layer: pp.LAYER.WGCLAD[0]
       clad_datatype: pp.LAYER.WGCLAD[1]
       bend_radius: 10
       cladding_offset: 3

    .. plot::
      :include-source:

      import pp

      c = pp.c.disk(radius=10, wrap_angle=3.14/4)
      pp.plotgds(c)

    """

    c = pc.Disk(
        pp.call_if_func(wg_strip, **kwargs),
        radius=radius,
        coupling_gap=gap,
        wrap_angle=wrap_angle,
        parity=parity,
        port=port,
        direction=direction,
    )

    return picwriter2component(c)


if __name__ == "__main__":
    import pp

    c = disk(wrap_angle=3.14 / 4)
    pp.show(c)
