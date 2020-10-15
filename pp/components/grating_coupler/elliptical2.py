import numpy as np
import picwriter.components as pc

from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component
from pp.port import deco_rename_ports
import pp


@deco_rename_ports
@pp.autoname
def grating_coupler_elliptical2(
    wgt=wg_strip,
    theta=np.pi / 4.0,
    length=30.0,
    taper_length=10.0,
    period=1.0,
    dutycycle=0.7,
    ridge=True,
    ridge_layers=(2, 0),
    teeth_list=None,
    port=(0, 0),
    direction="EAST",
    polarization="te",
    wavelength=1550,
    **kwargs
):
    """ Grating coupler

    Args:
        waveguide_template: object or function
        port (tuple): Cartesian coordinate of the input port
        direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
        theta (float): Angle of the waveguide.  Defaults to pi/4.0
        length (float): Length of the total grating coupler region, measured from the output port.  Defaults to 30.0
        taper_length (float): Length of the taper before the grating coupler.  Defaults to 10.0
        period (float): Grating period.  Defaults to 1.0
        dutycycle (float): dutycycle, determines the size of the 'gap' by dutycycle=(period-gap)/period.  Defaults to 0.7
        ridge (boolean): If True, adds another layer to the grating coupler that can be used for partial etched gratings
        ridge_layers (tuple): Tuple specifying the layer/datatype of the ridge region.  Defaults to (3,0)
        teeth_list (list): Can optionally pass a list of (gap, width) tuples to be used as the gap and teeth widths for irregularly spaced gratings.  For example, [(0.6, 0.2), (0.7, 0.3), ...] would be a gap of 0.6, then a tooth of width 0.2, then gap of 0.7 and tooth of 0.3, and so on.  Overrides *period*, *dutycycle*, and *length*.  Defaults to None.

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_elliptical2()
      pp.plotgds(c)

    """

    c = pc.GratingCoupler(
        pp.call_if_func(wg_strip, **kwargs),
        theta=theta,
        length=length,
        taper_length=taper_length,
        period=period,
        dutycycle=dutycycle,
        ridge=ridge,
        ridge_layers=ridge_layers,
        teeth_list=teeth_list,
        port=port,
        direction=direction,
    )

    c = picwriter2component(c)
    c.polarization = polarization
    c.wavelength = wavelength

    return c


if __name__ == "__main__":
    import pp

    c = grating_coupler_elliptical2(pins=True)
    print(c.ports)
    pp.show(c)
