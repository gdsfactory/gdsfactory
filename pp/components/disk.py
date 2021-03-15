from typing import Tuple

import numpy as np
import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component
from pp.types import ComponentFactory


@pp.cell
def disk(
    radius: float = 10.0,
    gap: float = 0.2,
    wrap_angle_deg: float = 180.0,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: ComponentFactory = wg_strip,
    **kwargs
) -> Component:
    """Disk Resonator

    Args:
       radius: disk resonator radius
       gap: Distance between the bus waveguide and resonator
       wrap_angle : Angle in degrees between 0 and 180
        determines how much the bus waveguide wraps along the resonator.
        0 corresponds to a straight bus waveguide,
        180 corresponds to a bus waveguide wrapped around half of the resonator.
       parity (1 or -1): If 1, resonator to left of bus waveguide, if -1 resonator to the right
       port (tuple): Cartesian coordinate of the input port (x1, y1)
       direction (string): Direction that the component will point *towards*, can be of type
        'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate object


    Other Parameters:
       wg_width: 0.5
       wg_layer: pp.LAYER.WG[0]
       wg_datatype: pp.LAYER.WG[1]
       clad_layer: pp.LAYER.WGCLAD[0]
       clad_datatype: pp.LAYER.WGCLAD[1]
       bend_radius: 10
       cladding_offset: 3

    """

    c = pc.Disk(
        pp.call_if_func(wg_strip, **kwargs),
        radius=radius,
        coupling_gap=gap,
        wrap_angle=wrap_angle_deg * np.pi / 180,
        parity=parity,
        port=port,
        direction=direction,
    )

    return picwriter2component(c)


if __name__ == "__main__":

    c = disk()
    c.show()
