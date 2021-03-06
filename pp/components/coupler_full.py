from typing import Callable, Tuple

import numpy as np
import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component


@pp.cell
def coupler_full(
    length: float = 40.0,
    gap: float = 0.5,
    dw: float = 0.1,
    angle: float = np.pi / 6,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: Callable = wg_strip,
    **kwargs
) -> Component:
    """ Adiabatic Full Coupler.
    Design based on asymmetric adiabatic full coupler designs, such as the one reported
    in 'Integrated Optic Adiabatic Devices on Silicon' by Y. Shani, et al
    (IEEE Journal of Quantum Electronics, Vol. 27, No. 3 March 1991).

    Region I is the first half of the input S-bend waveguide where the
    input waveguides widths taper by +dw and -dw,
    Region II is the second half of the S-bend waveguide with constant,
    unbalanced widths,
    Region III is the coupling region where the waveguides from unbalanced widths to
    balanced widths to reverse polarity unbalanced widths,
    Region IV is the fixed width waveguide that curves away from the coupling region,
    Region V is the final curve where the waveguides taper back to the regular width
    specified in the waveguide template.

    Args:
       length: Length of the coupling region.
       gap: Distance between the two waveguides.
       dw: Change in waveguide width. Top arm tapers to width - dw, bottom to width - dw.
       angle: Angle in radians at which the waveguide bends towards the coupling region.
       parity (integer -1 or 1): If -1, mirror-flips the structure so that the input port
        is actually the *bottom* port.
       port: Cartesian coordinate of the input port (AT TOP if parity=1, AT BOTTOM if parity=-1).
       direction: Direction that the component will point *towards*,
        can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians).
       waveguide_template: function that returns Picwriter WaveguideTemplate object

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

      c = pp.c.coupler_full(length=40, gap=0.2, dw=0.1)
      c.plot()

    """

    c = pc.FullCoupler(
        pp.call_if_func(waveguide_template, **kwargs),
        length=length,
        gap=gap,
        dw=dw,
        angle=angle,
        parity=parity,
        port=port,
        direction=direction,
    )

    return picwriter2component(c)


if __name__ == "__main__":

    c = coupler_full(length=40, gap=0.2, dw=0.1)
    c.show()
