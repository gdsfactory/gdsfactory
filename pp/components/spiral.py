from typing import Callable, Optional, Tuple

import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component
from pp.port import auto_rename_ports


@pp.cell
def spiral(
    width: float = 500.0,
    length: float = 10e3,
    spacing: Optional[float] = None,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "NORTH",
    waveguide_template: Callable = wg_strip,
    **kwargs
) -> Component:
    """ Picwriter Spiral

    Args:
       width (float): width of the spiral (i.e. distance between input/output ports)
       length (float): desired length of the waveguide (um)
       spacing (float): distance between parallel waveguides
       parity (int): If 1 spiral on right side, if -1 spiral on left side (mirror flip)
       port (tuple): Cartesian coordinate of the input port
       direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate object
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

      c = pp.c.spiral(width=500, length=10e3)
      pp.plotgds(c)

    """
    c = pc.Spiral(
        pp.call_if_func(waveguide_template, **kwargs),
        width=width,
        length=length,
        spacing=spacing,
        parity=parity,
        port=port,
        direction=direction,
    )
    # print(f'length = {length/1e4:.2f}cm')
    c = picwriter2component(c)
    c = auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = spiral(length=10e3, width=500, bend_radius=20)
    pp.show(c)
