from typing import Optional, Tuple

import numpy as np
import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import ComponentFactory


@gf.cell
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
    waveguide_template: ComponentFactory = strip,
    **kwargs
) -> Component:
    """Grating-Assisted Contra-Directional Coupler

    Args:
       length : Length of the coupling region.
       gap: Distance between the two straights.
       period: Period of the grating.
       dc: Duty cycle of the grating. Must be between 0 and 1.
       angle: in radians at which the straight bends towards the coupling region.
       width_top: Width of the top straight in the coupling region.
       width_bot: Width of the bottom straight in the coupling region.
       dw_top: Amplitude of the width variation on the top.  Default=gap/2.0.
       dw_bot: Amplitude of the width variation on the bottom.  Default=gap/2.0.
       input_bot: True makes the default input the bottom straight (rather than top)
       fins: If `True`, adds fins to the input/output straights.
        In this case a different template for the component must be specified.
        This feature is useful when performing electron-beam lithography
        and using different beam currents
        for fine features (helps to reduce stitching errors).
       fin_size: Specifies the x- and y-size of the `fins`. Defaults to 200 nm x 50 nm
       contradc_wgt:
       port_midpoint: Cartesian coordinate of the input port
        (AT TOP if input_bot=False, AT BOTTOM if input_bot=True).
       direction: Direction that the component will point *towards*,
        can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`,
        OR an angle (float, in radians).
       waveguide_template: Picwriter WaveguideTemplate function


    """

    c = pc.ContraDirectionalCoupler(
        gf.call_if_func(strip, **kwargs),
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

    component = gf.component_from.picwriter(c)
    gf.port.rename_ports_by_orientation(component)
    return component


if __name__ == "__main__":

    c = cdc()
    print(c.ports.keys())
    c.show()
