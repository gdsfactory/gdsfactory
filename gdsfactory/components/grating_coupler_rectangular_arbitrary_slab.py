from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.taper import taper_strip_to_slab150
from gdsfactory.types import ComponentFactory, Floats, Layer

_gaps = (0.2,) * 10
_widths = (0.5,) * 10


@gf.cell
def grating_coupler_rectangular_arbitrary_slab(
    gaps: Floats = _gaps,
    widths: Floats = _widths,
    wg_width: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 150.0,
    layer: Tuple[int, int] = gf.LAYER.WG,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentFactory = taper_strip_to_slab150,
    layer_slab: Optional[Layer] = gf.LAYER.SLAB150,
    slab_offset: float = 2.0,
) -> Component:
    r"""Grating coupler uniform (grating with rectangular shape not elliptical).
    Therefore it needs a longer taper.
    Grating teeth are straight instead of elliptical.

    Args:
        gaps: list of gaps
        widths: list of widths
        wg_width: input waveguide width
        width_grating:
        length_taper:
        layer: for grating teeth
        polarization: 'te' or 'tm'
        wavelength: in um
        taper: function
        layer_slab:
        slab_offset

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

    taper = taper(
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        w_slab2=width_grating + 2 * slab_offset,
    )

    taper_ref = c << taper

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    x0 = xi = taper_ref.xmax

    widths = gf.snap.snap_to_grid(widths)
    gaps = gf.snap.snap_to_grid(gaps)

    for width, gap in zip(widths, gaps):
        xi += gap + width / 2
        cgrating = c.add_ref(
            rectangle(
                size=[width, width_grating],
                layer=layer,
                port_type=None,
                centered=True,
            )
        )
        cgrating.x = gf.snap.snap_to_grid(xi)
        cgrating.y = 0
        xi += width / 2

    if layer_slab:
        slab = c << rectangle(
            size=(
                gf.snap.snap_to_grid(xi - x0) + slab_offset,
                width_grating + 2 * slab_offset,
            ),
            layer=layer_slab,
            port_type=None,
            centered=True,
        )
        slab.xmin = x0

    xport = np.round((xi + length_taper) / 2, 3)

    port_type = f"vertical_{polarization.lower()}"
    c.add_port(name=port_type, port_type=port_type, midpoint=(xport, 0), orientation=0)
    c.info.polarization = polarization
    c.info.wavelength = wavelength
    gf.asserts.grating_coupler(c)
    return c


if __name__ == "__main__":
    c = grating_coupler_rectangular_arbitrary_slab(slab_offset=2.0)
    print(c.ports)
    c.show()
