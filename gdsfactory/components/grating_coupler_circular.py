from typing import Optional

import numpy as np
import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import Coordinate, CrossSectionSpec, Floats, LayerSpec


@cell
def grating_coupler_circular(
    taper_angle: float = 30.0,
    taper_length: float = 10.0,
    length: float = 30.0,
    period: float = 1.0,
    fill_factor: float = 0.7,
    n_periods: int = 30,
    bias_gap: float = 0,
    port: Coordinate = (0.0, 0.0),
    layer_slab: Optional[LayerSpec] = None,
    layer_cladding: LayerSpec = "WGCLAD",
    gaps: Optional[Floats] = None,
    widths: Optional[Floats] = None,
    direction: str = "EAST",
    polarization: str = "te",
    wavelength: float = 1.55,
    fiber_marker_width: float = 11.0,
    fiber_marker_layer: Optional[LayerSpec] = "TE",
    cladding_offset: float = 2.0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Return circular Grating coupler.

    Args:
        taper_angle: taper flare angle in degrees.
        taper_length: Length of the taper before the grating coupler.
        length: total grating coupler length.
        period: Grating period.
        fill_factor: (period-gap)/period.
        n_periods: number of grating teeth.
        bias_gap: etch gap (um).
            Positive bias increases gap and reduces width to keep period constant.
        port: (x, y) for input port.
        layer_slab: slab layer for partial etched gratings.
        layer_cladding: for the cladding (using cladding_offset).
        gaps: optional gap list (um). Overrides period, fill_factor and n_periods.
        widths: optional width list (um). Overrides period, fill_factor and n_periods.
        direction: Direction that the component will point *towards*,
          can be of type NORTH, WEST, SOUTH, EAST, OR an angle (float, in radians).
        polarization: te or tm.
        wavelength: wavelength um.
        fiber_marker_width: (um).
        cladding_offset: (um).
        cross_section: for input waveguide port.
        kwargs: cross_section settings.


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
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer
    wg_width = x.width

    widths = widths or n_periods * [period * fill_factor]
    gaps = gaps or n_periods * [period * (1 - fill_factor)]

    gaps = np.array(gaps) + bias_gap
    widths = np.array(widths) - bias_gap
    teeth_list = list(zip(gaps, widths))

    layer = gf.get_layer(layer)
    layer_slab = gf.get_layer(layer_slab)
    layer_cladding = gf.get_layer(layer_cladding)

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
        ridge=bool(layer_slab),
        ridge_layers=layer_slab,
        teeth_list=teeth_list,
        port=port,
        direction=direction,
    )

    c = gf.read.from_picwriter(c)
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    c.ports["o1"].cross_section = x

    fiber_xoffset = np.round(c.center[0] + taper_length / 2, 3)

    if fiber_marker_layer:
        circle = gf.components.circle(
            radius=fiber_marker_width / 2, layer=fiber_marker_layer
        )
        circle_ref = c.add_ref(circle)
        circle_ref.movex(fiber_xoffset)

    c.add_port(
        name=f"vertical_{polarization.lower()}",
        midpoint=[fiber_xoffset, 0],
        width=fiber_marker_width,
        orientation=0,
        layer=fiber_marker_layer,
    )
    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


if __name__ == "__main__":
    # c = grating_coupler_circular(layer=(1, 0), period=1, fill_factor=0.7, bias_gap=0.1)
    c = grating_coupler_circular(
        layer=(1, 0),
        period=1,
        fill_factor=0.7,
        bias_gap=0.0,
        widths=[0.2] * 3,
        gaps=[0.5] * 3,
    )

    # c = gf.c.extend_ports(c)
    # print(len(c.name))
    # print(c.ports)
    c.show()
