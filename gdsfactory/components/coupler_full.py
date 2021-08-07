from typing import Tuple

import numpy as np
import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import ComponentFactory


@gf.cell
def coupler_full(
    length: float = 40.0,
    gap: float = 0.5,
    dw: float = 0.1,
    angle: float = np.pi / 6,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: ComponentFactory = strip,
    **kwargs
) -> Component:
    """Adiabatic Full Coupler.
    Design based on asymmetric adiabatic full coupler designs, such as the one reported
    in 'Integrated Optic Adiabatic Devices on Silicon' by Y. Shani, et al
    (IEEE Journal of Quantum Electronics, Vol. 27, No. 3 March 1991).

    Region I is the first half of the input S-bend straight where the
    input straights widths taper by +dw and -dw,
    Region II is the second half of the S-bend straight with constant,
    unbalanced widths,
    Region III is the coupling region where the straights from unbalanced widths to
    balanced widths to reverse polarity unbalanced widths,
    Region IV is the fixed width straight that curves away from the coupling region,
    Region V is the final curve where the straights taper back to the regular width
    specified in the straight template.

    Args:
       length: Length of the coupling region.
       gap: Distance between the two straights.
       dw: Change in straight width. Top arm tapers to width - dw, bottom to width - dw.
       angle: Angle in radians at which the straight bends towards the coupling region.
       parity (integer -1 or 1): If -1, mirror-flips the structure so that the input port
        is actually the *bottom* port.
       port: Cartesian coordinate of the input port (AT TOP if parity=1, AT BOTTOM if parity=-1).
       direction: Direction that the component will point *towards*,
        can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians).
       waveguide_template: function that returns Picwriter WaveguideTemplate object

    Other Parameters:
       wg_width: 0.5
       wg_layer: gf.LAYER.WG[0]
       wg_datatype: gf.LAYER.WG[1]
       clad_layer: gf.LAYER.WGCLAD[0]
       clad_datatype: gf.LAYER.WGCLAD[1]
       bend_radius: 10
       cladding_offset: 3


    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.coupler_full(length=40, gap=0.2, dw=0.1)
      c.plot()

    """

    c = pc.FullCoupler(
        gf.call_if_func(waveguide_template, **kwargs),
        length=length,
        gap=gap,
        dw=dw,
        angle=angle,
        parity=parity,
        port=port,
        direction=direction,
    )

    return gf.component_from.picwriter(c)


if __name__ == "__main__":

    c = coupler_full(length=40, gap=0.2, dw=0.1)
    c.show()
