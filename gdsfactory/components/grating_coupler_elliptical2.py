from typing import Optional

import numpy as np
import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.port import deco_rename_ports
from gdsfactory.types import ComponentFactory, Coordinate, Coordinates, Layer


@deco_rename_ports
@cell
def grating_coupler_elliptical2(
    wgt: ComponentFactory = strip,
    wg_width: float = 0.5,
    cladding_offset: float = 2.0,
    theta: float = np.pi / 4.0,
    length: float = 30.0,
    taper_length: float = 10.0,
    period: float = 1.0,
    dutycycle: float = 0.7,
    port: Coordinate = (0.0, 0.0),
    layer_ridge: Optional[Layer] = None,
    layer_core: Layer = gf.LAYER.WG,
    layer_cladding: Layer = gf.LAYER.WGCLAD,
    teeth_list: Optional[Coordinates] = None,
    direction: str = "EAST",
    polarization: str = "te",
    wavelength: float = 1.55,
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Layer = gf.LAYER.TE,
    **kwargs,
) -> Component:
    r"""Returns Grating coupler from Picwriter

    Args:
        wgt: waveguide_template object or function
        theta: Angle of the straight in rad.
        length: total grating coupler region.
        taper_length: Length of the taper before the grating coupler.
        period: Grating period.
        dutycycle: (period-gap)/period.
        port: Cartesian coordinate of the input port
        layer_ridge: for partial etched gratings
        layer_core: Tuple specifying the layer/datatype of the ridge region.
        layer_cladding: for the straight.
        teeth_list: (gap, width) tuples to be used as the gap and teeth widths
            for irregularly spaced gratings.
            For example, [(0.6, 0.2), (0.7, 0.3), ...] would be a gap of 0.6,
            then a tooth of width 0.2, then gap of 0.7 and tooth of 0.3, and so on.
            Overrides *period*, *dutycycle*, and *length*.  Defaults to None.
        direction: Direction that the component will point *towards*,
            can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`,
            OR an angle (float, in radians)
        polarization: te or tm
        wavelength: wavelength um

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.grating_coupler_elliptical2()
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
        gf.call_if_func(
            wgt,
            cladding_offset=cladding_offset,
            wg_width=wg_width,
            layer=layer_core,
            layer_cladding=layer_cladding,
            **kwargs,
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

    c = gf.read.picwriter(c)
    c.polarization = polarization
    c.wavelength = wavelength

    x = c.center[0] + taper_length / 2
    circle = gf.components.circle(
        radius=fiber_marker_width / 2, layer=fiber_marker_layer
    )
    circle_ref = c.add_ref(circle)
    circle_ref.movex(x)

    c.add_port(
        name=f"vertical_{polarization.lower()}",
        midpoint=[x, 0],
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
    )

    return c


if __name__ == "__main__":

    c = grating_coupler_elliptical2(length=31, taper_length=30)
    print(len(c.name))
    print(c.ports)
    c.show()
