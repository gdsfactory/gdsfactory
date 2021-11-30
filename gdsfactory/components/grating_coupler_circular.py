from typing import Optional

import numpy as np
import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import Coordinate, Coordinates, Floats, Layer


@cell
def grating_coupler_circular(
    taper_angle: float = 30.0,
    taper_length: float = 10.0,
    length: float = 30.0,
    period: float = 1.0,
    fill_factor: float = 0.7,
    port: Coordinate = (0.0, 0.0),
    layer: Layer = gf.LAYER.WG,
    layer_slab: Optional[Layer] = None,
    layer_cladding: Layer = gf.LAYER.WGCLAD,
    teeth_list: Optional[Coordinates] = None,
    direction: str = "EAST",
    polarization: str = "te",
    wavelength: float = 1.55,
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Optional[Layer] = gf.LAYER.TE,
    wg_width: float = 0.5,
    cladding_offset: float = 2.0,
) -> Component:
    r"""Returns Grating coupler from Picwriter

    Args:
        taper_angle: taper flare angle in degrees
        taper_length: Length of the taper before the grating coupler.
        length: total grating coupler length.
        period: Grating period.
        fill_factor: (period-gap)/period.
        port: Cartesian coordinate of the input port
        layer: waveguide layer
        layer_slab: slab layer for partial etched gratings
        layer_cladding: for the cladding (using cladding_offset)
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
        fiber_marker_width:
        wg_width: for input waveguide
        cladding_offset:


    .. code::

        side view
                      fiber
                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG  o1  ______________|

                      /
        top view     / |
                    /| |
                   / | |
                  /taper_angle
                 /_ _| |
        wg_width |   | |
                 \   | |
                  \  | |
                   \ | |
                    \| |
                 <-->
                taper_length
    """

    c = pc.GratingCoupler(
        gf.call_if_func(
            strip,
            cladding_offset=cladding_offset,
            wg_width=wg_width,
            layer=layer,
            layer_cladding=layer_cladding,
        ),
        theta=np.deg2rad(taper_angle),
        length=length,
        taper_length=taper_length,
        period=period,
        dutycycle=fill_factor,
        ridge=True if layer_slab else False,
        ridge_layers=layer_slab,
        teeth_list=teeth_list,
        port=port,
        direction=direction,
    )

    c = gf.read.from_picwriter(c)
    c.info.polarization = polarization
    c.info.wavelength = wavelength
    x = np.round(c.center[0] + taper_length / 2, 3)

    if fiber_marker_layer:
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

    c.auto_rename_ports()
    return c


_gap_width = tuple([0.1] * 10 + [0.5] * 10)


@cell
def grating_coupler_circular_arbitrary(teeth_list: Floats = _gap_width, **kwargs):
    """Returns grating coupler,
    teeth list is on a single list that starts with gap, width, gap ...

    Args:
        teeth_list: list of gaps and widths
        kwargs: for grating_coupler_circular
    """
    teeth_list = zip(teeth_list[::2], teeth_list[1::2])
    return grating_coupler_circular(teeth_list=list(teeth_list), **kwargs)


if __name__ == "__main__":
    # c = grating_coupler_circular_arbitrary(taper_length=30, layers_slab=((2,0), (3,0)))
    # c = grating_coupler_circular_arbitrary(taper_length=30, layer_slab=(2, 3))
    c = grating_coupler_circular(layer_slab=(3, 0))
    print(len(c.name))
    print(c.ports)
    c.show()
