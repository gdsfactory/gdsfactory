from typing import Optional

import numpy as np
import picwriter.components as pc

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.config import conf
from pp.picwriter2component import picwriter2component
from pp.port import deco_rename_ports
from pp.types import ComponentFactory, Coordinate, Layer


@deco_rename_ports
@cell
def grating_coupler_elliptical2(
    wgt: ComponentFactory = wg_strip,
    wg_width: float = 0.5,
    cladding_offset: float = conf.tech.cladding_offset,
    theta: float = np.pi / 4.0,
    length: float = 30.0,
    taper_length: float = 10.0,
    period: float = 1.0,
    dutycycle: float = 0.7,
    port: Coordinate = (0.0, 0.0),
    layer_ridge: Optional[Layer] = None,
    layer_core: Layer = pp.LAYER.WG,
    layer_cladding: Layer = pp.LAYER.WGCLAD,
    teeth_list: Optional[bool] = None,
    direction: str = "EAST",
    polarization: str = "te",
    wavelength_nm: str = 1550,
    **kwargs
) -> Component:
    r"""Returns Grating coupler from Picwriter

    Args:
        wgt: waveguide_template object or function
        theta: Angle of the waveguide in rad.
        length: Length of the total grating coupler region, measured from the output port.
        taper_length: Length of the taper before the grating coupler.
        period: Grating period.  Defaults to 1.0
        dutycycle: dutycycle, determines the size of the 'gap' by dutycycle=(period-gap)/period.
        port: Cartesian coordinate of the input port
        layer_ridge :  adds another layer to the grating coupler that can be used for partial etched gratings
        layer_core : Tuple specifying the layer/datatype of the ridge region.
        layer_cladding: for the waveguide.
        teeth_list: Can optionally pass a list of (gap, width) tuples to be used as the gap and teeth widths
            for irregularly spaced gratings.
            For example, [(0.6, 0.2), (0.7, 0.3), ...] would be a gap of 0.6,
            then a tooth of width 0.2, then gap of 0.7 and tooth of 0.3, and so on.
            Overrides *period*, *dutycycle*, and *length*.  Defaults to None.
        direction: Direction that the component will point *towards*,
            can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`,
            OR an angle (float, in radians)
        polarization: te or tm
        wavelength_nm: wavelength in nm

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_elliptical2()
      c.plot()

    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG->W0  ______________|

    """
    ridge = True if layer_ridge else False

    c = pc.GratingCoupler(
        pp.call_if_func(
            wgt,
            cladding_offset=cladding_offset,
            wg_width=wg_width,
            layer=layer_core,
            layer_cladding=layer_cladding,
            **kwargs
        ),
        theta=theta,
        length=length,
        taper_length=taper_length,
        period=period,
        dutycycle=1 - dutycycle,
        ridge=ridge,
        ridge_layers=layer_ridge,
        teeth_list=teeth_list,
        port=port,
        direction=direction,
    )

    c = picwriter2component(c)
    c.polarization = polarization
    c.wavelength = wavelength_nm
    return c


if __name__ == "__main__":

    c = grating_coupler_elliptical2()
    print(c.ports)
    c.show()
