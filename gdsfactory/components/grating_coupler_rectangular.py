from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory


@gf.cell
def grating_coupler_rectangular(
    n_periods: int = 20,
    period: float = 0.75,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    wg_width: float = 0.5,
    layer: Tuple[int, int] = gf.LAYER.WG,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentFactory = taper_function,
    layer_slab: Optional[Tuple[int, int]] = LAYER.SLAB150,
    slab_xmin: float = -1.0,
    slab_offset: float = 1.0,
) -> Component:
    r"""Grating coupler uniform (grating with rectangular shape not elliptical).
    Therefore it needs a longer taper.
    Grating teeth are straight.
    For a focusing grating take a look at grating_coupler_elliptical.


    Args:
        n_periods: number of grating teeth
        period: grating pitch
        fill_factor: ratio of grating width vs gap
        width_grating: 11
        length_taper: 150
        wg_width: input waveguide width
        layer: for grating teeth
        polarization: 'te' or 'tm'
        wavelength: in um
        taper: function
        layer_slab: layer that protects the slab under the grating
        slab_xmin: where 0 is at the start of the taper
        slab_offset: from edge of grating to edge of the slab

    .. code::

        side view
                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|


        top view     _________
                    /| | | | |
                   / | | | | |
                  /taper_angle
                 /_ _| | | | |
        wg_width |   | | | | |
                 \   | | | | |
                  \  | | | | |
                   \ | | | | |
                    \|_|_|_|_|
                 <-->
                taper_length
    """
    c = Component()
    taper_ref = c << taper_function(
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        layer=layer,
    )

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    x0 = taper_ref.xmax

    for i in range(n_periods):
        xsize = gf.snap.snap_to_grid(period * fill_factor)
        cgrating = c.add_ref(
            rectangle(size=[xsize, width_grating], layer=layer, port_type=None)
        )
        cgrating.xmin = gf.snap.snap_to_grid(x0 + i * period)
        cgrating.y = 0

    xport = np.round((x0 + cgrating.x) / 2, 3)

    port_type = f"vertical_{polarization.lower()}"
    c.add_port(name=port_type, port_type=port_type, midpoint=(xport, 0), orientation=0)
    c.info.polarization = polarization
    c.info.wavelength = wavelength
    gf.asserts.grating_coupler(c)

    if layer_slab:
        slab_xmin += length_taper
        slab_xsize = cgrating.xmax + slab_offset
        slab_ysize = c.ysize + 2 * slab_offset
        yslab = slab_ysize / 2
        c.add_polygon(
            [
                (slab_xmin, yslab),
                (slab_xsize, yslab),
                (slab_xsize, -yslab),
                (slab_xmin, -yslab),
            ],
            layer_slab,
        )
    return c


if __name__ == "__main__":
    # c = grating_coupler_rectangular(name='gcu', partial_etch=True)
    c = grating_coupler_rectangular()
    print(c.ports)
    c.show()
