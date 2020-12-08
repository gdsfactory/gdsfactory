from typing import Callable, Optional, Tuple

import numpy as np
import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component


@pp.cell
def cdc(
    length: float = 30.0,
    gap: float = 0.5,
    period: float = 0.220,
    dc: float = 0.5,
    angle: float = np.pi / 6.0,
    width_top: float = 2.0,
    width_bot: float = 0.75,
    input_bot: bool = False,
    dw_top: Optional[float] = None,
    dw_bot: Optional[float] = None,
    fins: bool = False,
    fin_size: Tuple[float, float] = (0.2, 0.05),
    contradc_wgt: None = None,
    port_midpoint: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: Callable = wg_strip,
    **kwargs
) -> Component:
    """Grating-Assisted Contra-Directional Coupler

    Args:
       length (float): Length of the coupling region.
       gap (float): Distance between the two waveguides.
       period (float): Period of the grating.
       dc (float): Duty cycle of the grating. Must be between 0 and 1.
       angle (float): Angle in radians (between 0 and pi/2) at which the waveguide bends towards the coupling region.  Default=pi/6.
       width_top (float): Width of the top waveguide in the coupling region.  Defaults to the WaveguideTemplate wg width.
       width_bot (float): Width of the bottom waveguide in the coupling region.  Defaults to the WaveguideTemplate wg width.
       dw_top (float): Amplitude of the width variation on the top.  Default=gap/2.0.
       dw_bot (float): Amplitude of the width variation on the bottom.  Default=gap/2.0.
       input_bot (boolean): If `True`, will make the default input the bottom waveguide (rather than the top).  Default=`False`
       fins (boolean): If `True`, adds fins to the input/output waveguides.  In this case a different template for the component must be specified.  This feature is useful when performing electron-beam lithography and using different beam currents for fine features (helps to reduce stitching errors).  Defaults to `False`
       fin_size ((x,y) Tuple): Specifies the x- and y-size of the `fins`.  Defaults to 200 nm x 50 nm
       contradc_wgt (WaveguideTemplate): If `fins` above is True, a WaveguideTemplate (contradc_wgt) must be specified.  This defines the layertype / datatype of the ContraDC (which will be separate from the input/output waveguides).  Defaults to `None`
       port_midpoint (tuple): Cartesian coordinate of the input port (AT TOP if input_bot=False, AT BOTTOM if input_bot=True).  Defaults to (0,0).
       direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians).  Defaults to 'EAST'.
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate object


    .. plot::
      :include-source:

      import pp

      c = pp.c.cdc()
      pp.plotgds(c)

    """

    c = pc.ContraDirectionalCoupler(
        pp.call_if_func(wg_strip, **kwargs),
        length=length,
        gap=gap,
        period=period,
        dc=dc,
        angle=angle,
        width_top=width_top,
        width_bot=width_bot,
        dw_top=dw_top,
        dw_bot=dw_bot,
        input_bot=input_bot,
        fins=fins,
        fin_size=fin_size,
        contradc_wgt=contradc_wgt,
        port=port_midpoint,
        direction=direction,
    )

    component = picwriter2component(c)
    pp.port.rename_ports_by_orientation(component)
    return component


if __name__ == "__main__":
    import pp

    c = cdc()
    print(c.ports.keys())
    pp.show(c)
